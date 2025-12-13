"""Variation Asset Contract Implementation.

Standardizes naming, placement, and metadata for variation assets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Constants
# =============================================================================


CONTRACT_VERSION = "1.0.0"

VALID_PRIORITIES = {"required", "recommended", "optional"}
VALID_SOURCE_HINTS = {"generate", "catalog", "import", None}
VALID_COLLISION_SHAPES = {"box", "sphere", "capsule", "convex", "trimesh"}

# Standard asset file names
ASSET_FILES = {
    "reference_image": "reference.png",
    "model_glb": "asset.glb",
    "model_usdz": "asset.usdz",
    "simready": "simready.usda",
    "metadata": "metadata.json",
}

# Physics defaults by semantic class
PHYSICS_DEFAULTS_BY_CLASS: Dict[str, Dict[str, Any]] = {
    "dish": {
        "mass_range_kg": [0.2, 0.8],
        "material_type": "ceramic",
        "collision_shape": "box",
        "graspable": True,
        "friction": 0.5,
        "restitution": 0.1,
    },
    "utensil": {
        "mass_range_kg": [0.02, 0.15],
        "material_type": "metal",
        "collision_shape": "capsule",
        "graspable": True,
        "friction": 0.3,
        "restitution": 0.2,
    },
    "bottle": {
        "mass_range_kg": [0.1, 1.5],
        "material_type": "glass",
        "collision_shape": "capsule",
        "graspable": True,
        "friction": 0.4,
        "restitution": 0.15,
    },
    "can": {
        "mass_range_kg": [0.3, 0.6],
        "material_type": "metal",
        "collision_shape": "capsule",
        "graspable": True,
        "friction": 0.35,
        "restitution": 0.2,
    },
    "box": {
        "mass_range_kg": [0.1, 2.0],
        "material_type": "cardboard",
        "collision_shape": "box",
        "graspable": True,
        "friction": 0.6,
        "restitution": 0.05,
    },
    "clothing": {
        "mass_range_kg": [0.1, 1.0],
        "material_type": "fabric",
        "collision_shape": "box",
        "graspable": True,
        "friction": 0.7,
        "restitution": 0.0,
    },
    "food": {
        "mass_range_kg": [0.05, 1.0],
        "material_type": "organic",
        "collision_shape": "convex",
        "graspable": True,
        "friction": 0.5,
        "restitution": 0.1,
    },
    "tool": {
        "mass_range_kg": [0.1, 2.0],
        "material_type": "metal",
        "collision_shape": "convex",
        "graspable": True,
        "friction": 0.4,
        "restitution": 0.15,
    },
    "container": {
        "mass_range_kg": [0.1, 0.5],
        "material_type": "plastic",
        "collision_shape": "box",
        "graspable": True,
        "friction": 0.5,
        "restitution": 0.1,
    },
    "object": {  # default
        "mass_range_kg": [0.1, 1.0],
        "material_type": "generic",
        "collision_shape": "box",
        "graspable": True,
        "friction": 0.5,
        "restitution": 0.1,
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VariationAssetSpec:
    """Specification for a single variation asset."""

    name: str
    category: str
    semantic_class: str
    description: str
    priority: str = "optional"
    source_hint: Optional[str] = "generate"
    example_variants: List[str] = field(default_factory=list)

    # Physics hints
    physics_hints: Dict[str, Any] = field(default_factory=dict)

    # Generation hints (used by variation-gen-job)
    material_hint: Optional[str] = None
    style_hint: Optional[str] = None
    generation_prompt_hint: Optional[str] = None

    # Status tracking
    generation_status: Optional[str] = None  # pending, success, failed
    reference_image_path: Optional[str] = None
    asset_path: Optional[str] = None
    simready_path: Optional[str] = None

    def __post_init__(self):
        # Validate and apply defaults
        if self.priority not in VALID_PRIORITIES:
            self.priority = "optional"
        if self.source_hint not in VALID_SOURCE_HINTS:
            self.source_hint = "generate"

        # Apply physics defaults if not provided
        if not self.physics_hints:
            self.physics_hints = dict(
                PHYSICS_DEFAULTS_BY_CLASS.get(
                    self.semantic_class,
                    PHYSICS_DEFAULTS_BY_CLASS["object"],
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariationAssetSpec":
        return cls(
            name=data.get("name", "unknown"),
            category=data.get("category", "other"),
            semantic_class=data.get("semantic_class", "object"),
            description=data.get("description", ""),
            priority=data.get("priority", "optional"),
            source_hint=data.get("source_hint"),
            example_variants=data.get("example_variants", []),
            physics_hints=data.get("physics_hints", {}),
            material_hint=data.get("material_hint"),
            style_hint=data.get("style_hint"),
            generation_prompt_hint=data.get("generation_prompt_hint"),
            generation_status=data.get("generation_status"),
            reference_image_path=data.get("reference_image_path"),
            asset_path=data.get("asset_path"),
            simready_path=data.get("simready_path"),
        )


@dataclass
class VariationManifest:
    """Manifest for all variation assets in a scene."""

    version: str = CONTRACT_VERSION
    scene_id: str = ""
    scene_type: str = "generic"
    environment_type: str = "generic"
    policies: List[str] = field(default_factory=list)
    assets: List[VariationAssetSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "environment_type": self.environment_type,
            "policies": self.policies,
            "assets": [a.to_dict() for a in self.assets],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariationManifest":
        assets = [
            VariationAssetSpec.from_dict(a) for a in data.get("assets", [])
        ]
        return cls(
            version=data.get("version", CONTRACT_VERSION),
            scene_id=data.get("scene_id", ""),
            scene_type=data.get("scene_type", "generic"),
            environment_type=data.get("environment_type", "generic"),
            policies=data.get("policies", []),
            assets=assets,
            metadata=data.get("metadata", {}),
        )

    def get_asset(self, name: str) -> Optional[VariationAssetSpec]:
        """Get asset by name."""
        for asset in self.assets:
            if asset.name == name:
                return asset
        return None

    def get_assets_by_category(self, category: str) -> List[VariationAssetSpec]:
        """Get all assets in a category."""
        return [a for a in self.assets if a.category == category]

    def get_assets_by_priority(self, priority: str) -> List[VariationAssetSpec]:
        """Get all assets with a specific priority."""
        return [a for a in self.assets if a.priority == priority]

    def get_pending_assets(self) -> List[VariationAssetSpec]:
        """Get assets that need generation."""
        return [
            a for a in self.assets
            if a.source_hint == "generate" and a.generation_status != "success"
        ]


# =============================================================================
# Utilities
# =============================================================================


def standardize_asset_name(name: str) -> str:
    """Standardize asset name to filesystem-safe format.

    Rules:
    - Lowercase
    - Replace spaces and special chars with underscores
    - Remove consecutive underscores
    - Max 64 characters
    """
    # Lowercase and replace spaces/special chars
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    # Truncate
    if len(name) > 64:
        name = name[:64].rstrip("_")

    return name or "unnamed_asset"


def get_asset_paths(
    base_dir: Path,
    asset_name: str,
) -> Dict[str, Path]:
    """Get standard paths for an asset.

    Args:
        base_dir: Base directory for variation assets
        asset_name: Standardized asset name

    Returns:
        Dict mapping file type to path
    """
    asset_dir = base_dir / asset_name
    return {
        "dir": asset_dir,
        "reference_image": asset_dir / ASSET_FILES["reference_image"],
        "model_glb": asset_dir / ASSET_FILES["model_glb"],
        "model_usdz": asset_dir / ASSET_FILES["model_usdz"],
        "simready": asset_dir / ASSET_FILES["simready"],
        "metadata": asset_dir / ASSET_FILES["metadata"],
    }


def validate_variation_manifest(
    manifest: VariationManifest,
) -> Tuple[bool, List[str]]:
    """Validate a variation manifest.

    Returns:
        (is_valid, list of warnings/errors)
    """
    issues = []

    # Check version
    if not manifest.version:
        issues.append("Missing version")

    # Check scene_id
    if not manifest.scene_id:
        issues.append("Missing scene_id")

    # Check assets
    if not manifest.assets:
        issues.append("No assets defined")

    seen_names = set()
    for asset in manifest.assets:
        # Check for duplicate names
        if asset.name in seen_names:
            issues.append(f"Duplicate asset name: {asset.name}")
        seen_names.add(asset.name)

        # Check required fields
        if not asset.name:
            issues.append("Asset missing name")
        if not asset.category:
            issues.append(f"Asset '{asset.name}' missing category")
        if not asset.semantic_class:
            issues.append(f"Asset '{asset.name}' missing semantic_class")

        # Check priority
        if asset.priority not in VALID_PRIORITIES:
            issues.append(
                f"Asset '{asset.name}' has invalid priority: {asset.priority}"
            )

        # Check physics hints
        physics = asset.physics_hints
        if physics:
            mass_range = physics.get("mass_range_kg")
            if mass_range:
                if not isinstance(mass_range, (list, tuple)) or len(mass_range) != 2:
                    issues.append(
                        f"Asset '{asset.name}' has invalid mass_range_kg format"
                    )
                elif mass_range[0] > mass_range[1]:
                    issues.append(
                        f"Asset '{asset.name}' has invalid mass_range_kg: min > max"
                    )

            collision_shape = physics.get("collision_shape")
            if collision_shape and collision_shape not in VALID_COLLISION_SHAPES:
                issues.append(
                    f"Asset '{asset.name}' has invalid collision_shape: {collision_shape}"
                )

    is_valid = len(issues) == 0
    return is_valid, issues


def create_variation_manifest(
    scene_id: str,
    environment_type: str,
    policies: List[str],
    assets: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> VariationManifest:
    """Create a standardized variation manifest.

    Args:
        scene_id: Scene identifier
        environment_type: Environment type (kitchen, office, etc.)
        policies: List of policy targets
        assets: List of asset specifications
        metadata: Optional additional metadata

    Returns:
        VariationManifest instance
    """
    asset_specs = []
    for asset_data in assets:
        # Standardize name
        name = standardize_asset_name(asset_data.get("name", ""))
        asset_data["name"] = name

        spec = VariationAssetSpec.from_dict(asset_data)
        asset_specs.append(spec)

    return VariationManifest(
        version=CONTRACT_VERSION,
        scene_id=scene_id,
        scene_type=environment_type,
        environment_type=environment_type,
        policies=policies,
        assets=asset_specs,
        metadata=metadata or {},
    )


# =============================================================================
# Contract Enforcer
# =============================================================================


class VariationAssetContract:
    """Enforces the variation asset contract across pipeline stages.

    Usage:
        contract = VariationAssetContract(scene_id, base_dir)

        # Create/load manifest
        manifest = contract.load_manifest()

        # Check asset status
        paths = contract.get_asset_paths("dirty_plate_01")
        status = contract.check_asset_status("dirty_plate_01")

        # Update after generation
        contract.mark_asset_generated("dirty_plate_01", success=True)
    """

    def __init__(
        self,
        scene_id: str,
        base_dir: Path,
        verbose: bool = True,
    ):
        self.scene_id = scene_id
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        self._manifest: Optional[VariationManifest] = None

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[VARIATION-CONTRACT] {msg}")

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "manifest.json"

    def load_manifest(self) -> VariationManifest:
        """Load manifest from file."""
        if self.manifest_path.is_file():
            data = json.loads(self.manifest_path.read_text())
            self._manifest = VariationManifest.from_dict(data)
            self.log(f"Loaded manifest with {len(self._manifest.assets)} assets")
        else:
            self._manifest = VariationManifest(scene_id=self.scene_id)
            self.log("Created new empty manifest")
        return self._manifest

    def save_manifest(self) -> None:
        """Save manifest to file."""
        if self._manifest is None:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self._manifest.to_dict(), indent=2)
        )
        self.log(f"Saved manifest to {self.manifest_path}")

    def get_asset_paths(self, asset_name: str) -> Dict[str, Path]:
        """Get paths for an asset."""
        return get_asset_paths(self.base_dir, asset_name)

    def check_asset_status(self, asset_name: str) -> Dict[str, Any]:
        """Check the status of an asset.

        Returns dict with:
        - has_reference_image: bool
        - has_model: bool
        - has_simready: bool
        - has_metadata: bool
        - complete: bool (all required files exist)
        """
        paths = self.get_asset_paths(asset_name)
        return {
            "has_reference_image": paths["reference_image"].is_file(),
            "has_model": paths["model_glb"].is_file() or paths["model_usdz"].is_file(),
            "has_simready": paths["simready"].is_file(),
            "has_metadata": paths["metadata"].is_file(),
            "complete": (
                paths["reference_image"].is_file()
                and (paths["model_glb"].is_file() or paths["model_usdz"].is_file())
                and paths["simready"].is_file()
            ),
        }

    def mark_asset_generated(
        self,
        asset_name: str,
        success: bool,
        reference_image_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark an asset as generated (or failed)."""
        if self._manifest is None:
            self.load_manifest()

        asset = self._manifest.get_asset(asset_name)
        if asset:
            asset.generation_status = "success" if success else "failed"
            if reference_image_path:
                asset.reference_image_path = reference_image_path
        self.save_manifest()

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the manifest."""
        if self._manifest is None:
            self.load_manifest()
        return validate_variation_manifest(self._manifest)

    def get_replicator_asset_paths(self) -> Dict[str, str]:
        """Get asset paths formatted for Replicator scripts.

        Returns dict mapping asset names to their simready USD paths
        (relative to scene root).
        """
        if self._manifest is None:
            self.load_manifest()

        paths = {}
        for asset in self._manifest.assets:
            status = self.check_asset_status(asset.name)
            if status["has_simready"]:
                paths[asset.name] = f"variation_assets/{asset.name}/simready.usda"
            elif status["has_model"]:
                if self.get_asset_paths(asset.name)["model_usdz"].is_file():
                    paths[asset.name] = f"variation_assets/{asset.name}/asset.usdz"
                else:
                    paths[asset.name] = f"variation_assets/{asset.name}/asset.glb"

        return paths
