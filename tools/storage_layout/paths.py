"""Storage Layout Paths Implementation.

Defines canonical paths and utilities for BlueprintPipeline storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STORAGE_VERSION = "1.0.0"


# =============================================================================
# Path Constants
# =============================================================================


# Relative paths within a scene directory
class Paths:
    """Canonical relative paths within a scene."""

    # Input
    INPUT_DIR = "input"
    INPUT_IMAGE = "input/room.jpg"

    # Segmentation/Inventory
    SEG_DIR = "seg"
    INVENTORY = "seg/inventory.json"

    # Assets
    ASSETS_DIR = "assets"
    MANIFEST = "assets/scene_manifest.json"
    LEGACY_MANIFEST = "assets/scene_assets.json"
    INTERACTIVE_DIR = "assets/interactive"

    # Layout
    LAYOUT_DIR = "layout"
    LAYOUT = "layout/scene_layout_scaled.json"

    # USD
    USD_DIR = "usd"
    SCENE_USD = "usd/scene.usda"

    # Replicator
    REPLICATOR_DIR = "replicator"
    PLACEMENT_REGIONS = "replicator/placement_regions.usda"
    BUNDLE_METADATA = "replicator/bundle_metadata.json"
    POLICIES_DIR = "replicator/policies"
    VARIATION_MANIFEST = "replicator/variation_assets/manifest.json"

    # Variation Assets
    VARIATION_ASSETS_DIR = "variation_assets"

    # Isaac Lab
    ISAAC_LAB_DIR = "isaac_lab"
    ENV_CFG = "isaac_lab/env_cfg.py"
    TRAIN_CFG = "isaac_lab/train_cfg.yaml"
    RANDOMIZATIONS = "isaac_lab/randomizations.py"
    REWARD_FUNCTIONS = "isaac_lab/reward_functions.py"


@dataclass
class ScenePaths:
    """Resolved paths for a scene."""

    root: Path
    scene_id: str

    # Input
    input_dir: Path
    input_image: Path

    # Segmentation
    seg_dir: Path
    inventory: Path

    # Assets
    assets_dir: Path
    manifest: Path
    legacy_manifest: Path
    interactive_dir: Path

    # Layout
    layout_dir: Path
    layout: Path

    # USD
    usd_dir: Path
    scene_usd: Path

    # Replicator
    replicator_dir: Path
    placement_regions: Path
    bundle_metadata: Path
    policies_dir: Path
    variation_manifest: Path

    # Variation Assets
    variation_assets_dir: Path

    # Isaac Lab
    isaac_lab_dir: Path
    env_cfg: Path
    train_cfg: Path

    def get_object_dir(self, obj_id: str) -> Path:
        """Get directory for a specific object."""
        return self.assets_dir / f"obj_{obj_id}"

    def get_interactive_dir(self, obj_id: str) -> Path:
        """Get interactive asset directory for an object."""
        return self.interactive_dir / f"obj_{obj_id}"

    def get_variation_asset_dir(self, asset_name: str) -> Path:
        """Get directory for a variation asset."""
        return self.variation_assets_dir / asset_name

    def get_policy_script(self, policy_id: str) -> Path:
        """Get path for a policy script."""
        return self.policies_dir / f"{policy_id}.py"

    def get_task_file(self, policy_id: str) -> Path:
        """Get path for a task file."""
        return self.isaac_lab_dir / f"task_{policy_id}.py"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary of string paths."""
        return {
            "root": str(self.root),
            "scene_id": self.scene_id,
            "manifest": str(self.manifest),
            "layout": str(self.layout),
            "scene_usd": str(self.scene_usd),
            "replicator_dir": str(self.replicator_dir),
            "isaac_lab_dir": str(self.isaac_lab_dir),
        }


def get_scene_paths(root: Path, scene_id: str) -> ScenePaths:
    """Get all paths for a scene.

    Args:
        root: GCS mount root (e.g., /mnt/gcs)
        scene_id: Scene identifier

    Returns:
        ScenePaths with all resolved paths
    """
    scene_root = root / "scenes" / scene_id

    return ScenePaths(
        root=scene_root,
        scene_id=scene_id,
        # Input
        input_dir=scene_root / Paths.INPUT_DIR,
        input_image=scene_root / Paths.INPUT_IMAGE,
        # Segmentation
        seg_dir=scene_root / Paths.SEG_DIR,
        inventory=scene_root / Paths.INVENTORY,
        # Assets
        assets_dir=scene_root / Paths.ASSETS_DIR,
        manifest=scene_root / Paths.MANIFEST,
        legacy_manifest=scene_root / Paths.LEGACY_MANIFEST,
        interactive_dir=scene_root / Paths.INTERACTIVE_DIR,
        # Layout
        layout_dir=scene_root / Paths.LAYOUT_DIR,
        layout=scene_root / Paths.LAYOUT,
        # USD
        usd_dir=scene_root / Paths.USD_DIR,
        scene_usd=scene_root / Paths.SCENE_USD,
        # Replicator
        replicator_dir=scene_root / Paths.REPLICATOR_DIR,
        placement_regions=scene_root / Paths.PLACEMENT_REGIONS,
        bundle_metadata=scene_root / Paths.BUNDLE_METADATA,
        policies_dir=scene_root / Paths.POLICIES_DIR,
        variation_manifest=scene_root / Paths.VARIATION_MANIFEST,
        # Variation Assets
        variation_assets_dir=scene_root / Paths.VARIATION_ASSETS_DIR,
        # Isaac Lab
        isaac_lab_dir=scene_root / Paths.ISAAC_LAB_DIR,
        env_cfg=scene_root / Paths.ENV_CFG,
        train_cfg=scene_root / Paths.TRAIN_CFG,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_asset_path(root: Path, scene_id: str, obj_id: str, filename: str = "asset.glb") -> Path:
    """Get path for an object asset file."""
    return root / "scenes" / scene_id / "assets" / f"obj_{obj_id}" / filename


def get_manifest_path(root: Path, scene_id: str) -> Path:
    """Get path for the canonical manifest."""
    return root / "scenes" / scene_id / Paths.MANIFEST


def get_layout_path(root: Path, scene_id: str) -> Path:
    """Get path for the scene layout."""
    return root / "scenes" / scene_id / Paths.LAYOUT


def get_usd_path(root: Path, scene_id: str) -> Path:
    """Get path for the scene USD."""
    return root / "scenes" / scene_id / Paths.SCENE_USD


def get_replicator_path(root: Path, scene_id: str) -> Path:
    """Get path for the Replicator directory."""
    return root / "scenes" / scene_id / Paths.REPLICATOR_DIR


def get_isaac_lab_path(root: Path, scene_id: str) -> Path:
    """Get path for the Isaac Lab directory."""
    return root / "scenes" / scene_id / Paths.ISAAC_LAB_DIR


# =============================================================================
# Validation
# =============================================================================


def validate_scene_structure(
    root: Path,
    scene_id: str,
    require_all: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """Validate scene directory structure.

    Args:
        root: GCS mount root
        scene_id: Scene identifier
        require_all: If True, all files must exist (strict mode)

    Returns:
        Tuple of (is_valid, missing_required, missing_optional)
    """
    paths = get_scene_paths(root, scene_id)

    required = [
        (paths.manifest, "scene_manifest.json"),
        (paths.layout, "scene_layout_scaled.json"),
        (paths.scene_usd, "scene.usda"),
    ]

    optional = [
        (paths.inventory, "inventory.json"),
        (paths.placement_regions, "placement_regions.usda"),
        (paths.env_cfg, "env_cfg.py"),
    ]

    missing_required = []
    missing_optional = []

    for path, name in required:
        if not path.is_file():
            missing_required.append(name)

    for path, name in optional:
        if not path.is_file():
            missing_optional.append(name)

    is_valid = len(missing_required) == 0
    if require_all:
        is_valid = is_valid and len(missing_optional) == 0

    return is_valid, missing_required, missing_optional


# =============================================================================
# Storage Layout Class
# =============================================================================


class StorageLayout:
    """Utility class for managing storage layout.

    Usage:
        layout = StorageLayout("/mnt/gcs", "scene_123")
        manifest_path = layout.manifest
        obj_dir = layout.get_object_dir("refrigerator")
    """

    def __init__(self, root: Path, scene_id: str):
        self.root = Path(root)
        self.scene_id = scene_id
        self._paths = get_scene_paths(self.root, scene_id)

    @property
    def paths(self) -> ScenePaths:
        """Get all scene paths."""
        return self._paths

    @property
    def scene_root(self) -> Path:
        """Get scene root directory."""
        return self._paths.root

    @property
    def manifest(self) -> Path:
        """Get manifest path."""
        return self._paths.manifest

    @property
    def layout(self) -> Path:
        """Get layout path."""
        return self._paths.layout

    @property
    def scene_usd(self) -> Path:
        """Get scene USD path."""
        return self._paths.scene_usd

    def get_object_dir(self, obj_id: str) -> Path:
        """Get directory for an object."""
        return self._paths.get_object_dir(obj_id)

    def get_interactive_dir(self, obj_id: str) -> Path:
        """Get interactive asset directory."""
        return self._paths.get_interactive_dir(obj_id)

    def get_variation_asset_dir(self, asset_name: str) -> Path:
        """Get variation asset directory."""
        return self._paths.get_variation_asset_dir(asset_name)

    def ensure_directories(self) -> None:
        """Create all required directories."""
        dirs = [
            self._paths.input_dir,
            self._paths.seg_dir,
            self._paths.assets_dir,
            self._paths.layout_dir,
            self._paths.usd_dir,
            self._paths.replicator_dir,
            self._paths.policies_dir,
            self._paths.variation_assets_dir,
            self._paths.isaac_lab_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def validate(self, require_all: bool = False) -> Tuple[bool, List[str], List[str]]:
        """Validate the storage structure."""
        return validate_scene_structure(self.root, self.scene_id, require_all)

    def get_gcs_uri(self, bucket: str, path: Path) -> str:
        """Convert local path to GCS URI."""
        rel_path = path.relative_to(self.root)
        return f"gs://{bucket}/{rel_path}"
