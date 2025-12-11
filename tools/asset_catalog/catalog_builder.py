"""
Asset Catalog Builder - Indexes NVIDIA asset packs for search and retrieval.

This module crawls asset pack directories and extracts metadata including:
- File paths and types
- Bounding box dimensions
- Variant sets
- Materials
- SimReady metadata (physics, semantics)
- Tags inferred from paths and LLM analysis

The catalog is stored as JSON/Parquet for efficient querying.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class AssetEntry:
    """Represents a single asset in the catalog."""

    asset_id: str
    relative_path: str
    file_type: str
    category: str
    subcategory: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    dimensions: Optional[dict[str, float]] = None
    variant_sets: list[dict[str, Any]] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    simready_metadata: Optional[dict[str, Any]] = None
    default_prim: Optional[str] = None
    thumbnail_path: Optional[str] = None
    embedding_id: Optional[str] = None


@dataclass
class AssetCatalog:
    """Complete catalog of an asset pack."""

    pack_name: str
    display_name: str
    version_hash: str
    indexed_at: str
    total_assets: int
    categories: list[str]
    assets: list[AssetEntry]


class AssetCatalogBuilder:
    """
    Builds searchable catalogs from NVIDIA asset packs.

    Usage:
        builder = AssetCatalogBuilder(
            pack_path="/path/to/ResidentialAssetsPack",
            pack_name="ResidentialAssetsPack"
        )
        catalog = builder.build()
        builder.save(catalog, "asset_index.json")
    """

    # Common category patterns in NVIDIA packs
    CATEGORY_PATTERNS = {
        "furniture": ["chair", "table", "sofa", "couch", "bed", "dresser", "desk", "shelf", "cabinet", "bookshelf"],
        "appliance": [
            "refrigerator",
            "fridge",
            "oven",
            "stove",
            "microwave",
            "dishwasher",
            "washer",
            "dryer",
            "vent",
            "hood",
            "venthood",
            "rangehood",
        ],
        "lighting": ["lamp", "light", "fixture", "chandelier", "sconce"],
        "decor": ["vase", "plant", "art", "frame", "mirror", "rug", "curtain"],
        "electronics": ["tv", "television", "monitor", "computer", "phone", "speaker"],
        "kitchen": ["pot", "pan", "dish", "plate", "cup", "mug", "bowl", "utensil", "fork", "knife", "spoon"],
        "bathroom": ["toilet", "sink", "bathtub", "shower", "towel"],
        "storage": ["box", "container", "basket", "bin", "crate", "pallet", "tote"],
        "industrial": ["rack", "racking", "conveyor", "forklift", "equipment"],
    }

    def __init__(self, pack_path: str, pack_name: str, display_name: Optional[str] = None):
        self.pack_path = Path(pack_path)
        self.pack_name = pack_name
        self.display_name = display_name or pack_name
        self._has_usd = self._check_usd_available()

        if self._has_usd:
            from pxr import Usd, UsdGeom

            self.Usd = Usd
            self.UsdGeom = UsdGeom

    def _check_usd_available(self) -> bool:
        """Check if OpenUSD is available."""
        try:
            from pxr import Usd  # noqa: F401

            return True
        except ImportError:
            return False

    def build(self, generate_thumbnails: bool = False) -> AssetCatalog:
        """
        Build the asset catalog by crawling the pack directory.

        Args:
            generate_thumbnails: Whether to generate thumbnail images

        Returns:
            AssetCatalog with all indexed assets
        """
        assets = []
        categories = set()

        # Find all USD files
        usd_files = self._find_usd_files()

        for usd_file in usd_files:
            entry = self._process_asset(usd_file)
            if entry:
                assets.append(entry)
                categories.add(entry.category)
                if entry.subcategory:
                    categories.add(entry.subcategory)

        # Generate version hash
        version_hash = self._compute_version_hash(assets)

        return AssetCatalog(
            pack_name=self.pack_name,
            display_name=self.display_name,
            version_hash=version_hash,
            indexed_at=datetime.utcnow().isoformat() + "Z",
            total_assets=len(assets),
            categories=sorted(list(categories)),
            assets=assets,
        )

    def _find_usd_files(self) -> list[Path]:
        """Find all USD files in the pack."""
        usd_extensions = [".usd", ".usda", ".usdc", ".usdz"]
        files = []

        for ext in usd_extensions:
            files.extend(self.pack_path.rglob(f"*{ext}"))

        # Filter out intermediate/reference files
        filtered = []
        for f in files:
            # Skip files in common non-asset directories
            parts = f.parts
            if any(skip in parts for skip in ["materials", "textures", "renders", ".backup"]):
                continue
            filtered.append(f)

        return sorted(filtered)

    def _process_asset(self, usd_path: Path) -> Optional[AssetEntry]:
        """Process a single USD file and extract metadata."""
        try:
            relative_path = str(usd_path.relative_to(self.pack_path))
            asset_id = self._generate_asset_id(relative_path)

            # Infer category from path
            category, subcategory = self._infer_category(relative_path)

            # Extract tags from path
            tags = self._extract_tags(relative_path)

            # Create basic entry
            entry = AssetEntry(
                asset_id=asset_id,
                relative_path=relative_path,
                file_type=usd_path.suffix,
                category=category,
                subcategory=subcategory,
                display_name=self._format_display_name(usd_path.stem),
                tags=tags,
            )

            # Extract USD metadata if available
            if self._has_usd:
                self._extract_usd_metadata(usd_path, entry)

            return entry

        except Exception as e:  # pragma: no cover - filesystem dependent
            print(f"Warning: Failed to process {usd_path}: {e}")
            return None

    def _extract_usd_metadata(self, usd_path: Path, entry: AssetEntry) -> None:
        """Extract metadata from USD file using OpenUSD."""
        try:
            stage = self.Usd.Stage.Open(str(usd_path))
            if not stage:
                return

            # Get default prim
            default_prim = stage.GetDefaultPrim()
            if default_prim:
                entry.default_prim = default_prim.GetPath().pathString

                # Get bounding box
                bbox_cache = self.UsdGeom.BBoxCache(self.Usd.TimeCode.Default(), [self.UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeWorldBound(default_prim)
                if bbox:
                    range_box = bbox.ComputeAlignedRange()
                    size = range_box.GetSize()
                    entry.dimensions = {
                        "width": float(size[0]),
                        "depth": float(size[1]) if len(size) > 1 else 0.0,
                        "height": float(size[2]) if len(size) > 2 else 0.0,
                    }

            # Get variant sets
            root = stage.GetPseudoRoot()
            for prim in self.Usd.PrimRange(root):
                vsets = prim.GetVariantSets()
                for vset_name in vsets.GetNames():
                    vset = vsets.GetVariantSet(vset_name)
                    variants = vset.GetVariantNames()
                    if variants:
                        entry.variant_sets.append({
                            "name": vset_name,
                            "variants": list(variants),
                            "default": vset.GetVariantSelection(),
                        })

            # Check for SimReady metadata
            entry.simready_metadata = self._extract_simready_metadata(stage)

            # Get materials
            entry.materials = self._extract_materials(stage)

        except Exception as e:  # pragma: no cover - USD extraction is optional
            print(f"Warning: USD metadata extraction failed for {usd_path}: {e}")

    def _extract_simready_metadata(self, stage: Any) -> dict[str, Any]:
        """Extract SimReady-specific metadata from stage."""
        metadata = {
            "is_simready": False,
            "physics_ready": False,
            "has_colliders": False,
            "semantic_labels": [],
            "articulations": [],
        }

        try:
            from pxr import UsdPhysics

            root = stage.GetPseudoRoot()
            for prim in self.Usd.PrimRange(root):
                # Check for physics
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    metadata["has_colliders"] = True
                    metadata["physics_ready"] = True

                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    metadata["physics_ready"] = True

                # Check for joints (articulations)
                if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                    metadata["articulations"].append({
                        "joint_name": prim.GetName(),
                        "joint_type": "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "prismatic",
                    })

                # Check for semantic labels
                semantic_class = prim.GetAttribute("semantic:class")
                if semantic_class and semantic_class.Get():
                    metadata["semantic_labels"].append(semantic_class.Get())

            # Mark as SimReady if it has physics or semantics
            if metadata["physics_ready"] or metadata["semantic_labels"]:
                metadata["is_simready"] = True

        except ImportError:
            pass

        return metadata

    def _extract_materials(self, stage: Any) -> list[str]:
        """Extract material names from stage."""
        materials = []
        try:
            from pxr import UsdShade

            root = stage.GetPseudoRoot()
            for prim in self.Usd.PrimRange(root):
                if prim.IsA(UsdShade.Material):
                    materials.append(prim.GetName())
        except ImportError:
            pass
        return materials

    def _generate_asset_id(self, relative_path: str) -> str:
        """Generate unique asset ID from path."""
        return hashlib.md5(relative_path.encode()).hexdigest()[:12]

    def _infer_category(self, path: str) -> tuple[str, Optional[str]]:
        """Infer category from file path."""
        path_lower = path.lower()

        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if pattern in path_lower:
                    # Try to find subcategory
                    for subcat in patterns:
                        if subcat in path_lower and subcat != pattern:
                            return category, subcat
                    return category, None

        # Fallback: use first directory as category
        parts = Path(path).parts
        if len(parts) > 1:
            return parts[0].lower(), None

        return "uncategorized", None

    def _extract_tags(self, path: str) -> list[str]:
        """Extract searchable tags from path."""
        # Split path into components and clean them
        parts = Path(path).parts[:-1]  # Exclude filename
        stem = Path(path).stem

        tags = []

        # Add path components as tags
        for part in parts:
            cleaned = part.lower().replace("_", " ").replace("-", " ")
            tags.extend(cleaned.split())

        # Add filename parts as tags
        stem_cleaned = stem.lower().replace("_", " ").replace("-", " ")
        tags.extend(stem_cleaned.split())

        # Remove duplicates and common words
        stop_words = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "usd", "usda", "usdc"}
        tags = list(set(t for t in tags if t not in stop_words and len(t) > 1))

        return sorted(tags)

    def _format_display_name(self, stem: str) -> str:
        """Format file stem as display name."""
        # Replace underscores/hyphens with spaces and title case
        name = stem.replace("_", " ").replace("-", " ")
        return " ".join(word.capitalize() for word in name.split())

    def _compute_version_hash(self, assets: list[AssetEntry]) -> str:
        """Compute a hash representing the catalog version."""
        content = json.dumps([{"id": a.asset_id, "path": a.relative_path} for a in assets], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self, catalog: AssetCatalog, output_path: str) -> None:
        """Save catalog to JSON file."""
        data = {
            "version": "1.0.0",
            "pack_info": {
                "name": catalog.pack_name,
                "display_name": catalog.display_name,
                "version_hash": catalog.version_hash,
                "indexed_at": catalog.indexed_at,
                "total_assets": catalog.total_assets,
                "categories": catalog.categories,
            },
            "assets": [
                {
                    "asset_id": a.asset_id,
                    "relative_path": a.relative_path,
                    "file_type": a.file_type,
                    "category": a.category,
                    "subcategory": a.subcategory,
                    "display_name": a.display_name,
                    "description": a.description,
                    "tags": a.tags,
                    "dimensions": a.dimensions,
                    "variant_sets": a.variant_sets,
                    "materials": a.materials,
                    "simready_metadata": a.simready_metadata,
                    "default_prim": a.default_prim,
                    "thumbnail_path": a.thumbnail_path,
                    "embedding_id": a.embedding_id,
                }
                for a in catalog.assets
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(catalog_path: str) -> AssetCatalog:
        """Load catalog from JSON file."""
        with open(catalog_path) as f:
            data = json.load(f)

        pack_info = data["pack_info"]
        assets = [
            AssetEntry(
                asset_id=a["asset_id"],
                relative_path=a["relative_path"],
                file_type=a["file_type"],
                category=a["category"],
                subcategory=a.get("subcategory"),
                display_name=a.get("display_name"),
                description=a.get("description"),
                tags=a.get("tags", []),
                dimensions=a.get("dimensions"),
                variant_sets=a.get("variant_sets", []),
                materials=a.get("materials", []),
                simready_metadata=a.get("simready_metadata"),
                default_prim=a.get("default_prim"),
                thumbnail_path=a.get("thumbnail_path"),
                embedding_id=a.get("embedding_id"),
            )
            for a in data["assets"]
        ]

        return AssetCatalog(
            pack_name=pack_info["name"],
            display_name=pack_info["display_name"],
            version_hash=pack_info["version_hash"],
            indexed_at=pack_info["indexed_at"],
            total_assets=pack_info["total_assets"],
            categories=pack_info["categories"],
            assets=assets,
        )
