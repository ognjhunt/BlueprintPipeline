"""
Asset Index Builder for Genie Sim 3.0.

Creates a Genie Sim compatible asset index from BlueprintPipeline assets,
including semantic descriptions for RAG retrieval and optional embeddings.

Genie Sim Asset Index stores:
    - USD paths
    - Collision hulls
    - Mass properties
    - Texture variants
    - Semantic descriptions (for RAG retrieval)
    - 2048-dim embeddings (optional, requires embedding model)

References:
    - Genie Sim 3.0 Paper: https://arxiv.org/html/2601.02078v1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as url_error
from urllib import request as url_request

import numpy as np


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class GenieSimMaterial:
    """Material properties for physics simulation."""

    friction: float = 0.5
    restitution: float = 0.1
    density: float = 1000.0  # kg/m^3

    def to_dict(self) -> Dict[str, float]:
        return {
            "friction": self.friction,
            "restitution": self.restitution,
            "density": self.density,
        }


@dataclass
class GenieSimAsset:
    """A single asset in the Genie Sim asset index."""

    asset_id: str
    usd_path: str
    semantic_description: str
    categories: List[str]

    # Physics properties
    mass: Optional[float] = None
    material: GenieSimMaterial = field(default_factory=GenieSimMaterial)
    collision_hull_path: Optional[str] = None

    # Variants
    texture_variants: Dict[str, str] = field(default_factory=dict)

    # Embedding (2048-dim for QWEN, generated separately)
    embedding: Optional[List[float]] = None

    # Licensing/provenance
    commercial_ok: bool = True
    license: str = "proprietary"
    source: str = "blueprintpipeline"

    # BlueprintPipeline metadata
    bp_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "asset_id": self.asset_id,
            "usd_path": self.usd_path,
            "semantic_description": self.semantic_description,
            "categories": self.categories,
            "mass": self.mass,
            "material": self.material.to_dict(),
            "collision_hull_path": self.collision_hull_path,
            "texture_variants": self.texture_variants,
            "commercial_ok": self.commercial_ok,
            "license": self.license,
            "source": self.source,
            "bp_metadata": self.bp_metadata,
        }

        if self.embedding:
            result["embedding"] = self.embedding

        return result


@dataclass
class GenieSimAssetIndex:
    """Complete Genie Sim asset index."""

    assets: List[GenieSimAsset]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": [a.to_dict() for a in self.assets],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save asset index to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    def get_asset(self, asset_id: str) -> Optional[GenieSimAsset]:
        """Get asset by ID."""
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        return None

    def filter_commercial(self) -> "GenieSimAssetIndex":
        """Return a new index with only commercially-usable assets."""
        return GenieSimAssetIndex(
            assets=[a for a in self.assets if a.commercial_ok],
            metadata={**self.metadata, "filtered": "commercial_only"},
        )


# =============================================================================
# Category Mapping
# =============================================================================

# Maps BlueprintPipeline categories to Genie Sim categories
CATEGORY_MAPPING = {
    # Kitchen
    "cup": ["kitchen", "container", "graspable"],
    "mug": ["kitchen", "container", "graspable"],
    "plate": ["kitchen", "flatware", "graspable"],
    "bowl": ["kitchen", "container", "graspable"],
    "utensil": ["kitchen", "tool", "graspable"],
    "bottle": ["kitchen", "container", "graspable"],
    "pot": ["kitchen", "cookware", "graspable"],
    "pan": ["kitchen", "cookware", "graspable"],
    "microwave": ["kitchen", "appliance", "articulated"],
    "refrigerator": ["kitchen", "appliance", "articulated"],
    "oven": ["kitchen", "appliance", "articulated"],
    "dishwasher": ["kitchen", "appliance", "articulated"],
    "sink": ["kitchen", "fixture", "static"],
    "faucet": ["kitchen", "fixture", "interactive"],
    "countertop": ["kitchen", "surface", "static"],
    "cabinet": ["kitchen", "storage", "articulated"],

    # Warehouse
    "box": ["warehouse", "container", "graspable"],
    "package": ["warehouse", "container", "graspable"],
    "carton": ["warehouse", "container", "graspable"],
    "tote": ["warehouse", "container", "graspable"],
    "pallet": ["warehouse", "support", "static"],
    "shelf": ["warehouse", "storage", "static"],
    "rack": ["warehouse", "storage", "static"],
    "conveyor": ["warehouse", "equipment", "static"],

    # Office
    "desk": ["office", "furniture", "static"],
    "chair": ["office", "furniture", "movable"],
    "monitor": ["office", "equipment", "static"],
    "keyboard": ["office", "equipment", "graspable"],
    "mouse": ["office", "equipment", "graspable"],
    "phone": ["office", "equipment", "graspable"],
    "book": ["office", "item", "graspable"],
    "pen": ["office", "item", "graspable"],
    "drawer": ["office", "storage", "articulated"],
    "filing_cabinet": ["office", "storage", "articulated"],

    # General furniture
    "door": ["furniture", "access", "articulated"],
    "window": ["furniture", "fixture", "articulated"],
    "table": ["furniture", "surface", "static"],
    "couch": ["furniture", "seating", "static"],
    "bed": ["furniture", "rest", "static"],

    # Default
    "object": ["general", "item", "graspable"],
    "unknown": ["general", "unknown"],
}

# Material type to physics properties
MATERIAL_PHYSICS = {
    "metal": GenieSimMaterial(friction=0.4, restitution=0.2, density=7800.0),
    "plastic": GenieSimMaterial(friction=0.5, restitution=0.3, density=1200.0),
    "wood": GenieSimMaterial(friction=0.6, restitution=0.15, density=700.0),
    "glass": GenieSimMaterial(friction=0.3, restitution=0.1, density=2500.0),
    "ceramic": GenieSimMaterial(friction=0.5, restitution=0.1, density=2300.0),
    "fabric": GenieSimMaterial(friction=0.7, restitution=0.05, density=500.0),
    "rubber": GenieSimMaterial(friction=0.9, restitution=0.6, density=1100.0),
    "paper": GenieSimMaterial(friction=0.5, restitution=0.05, density=700.0),
    "cardboard": GenieSimMaterial(friction=0.5, restitution=0.1, density=300.0),
    "default": GenieSimMaterial(friction=0.5, restitution=0.1, density=1000.0),
}


# =============================================================================
# Semantic Description Generator
# =============================================================================


class SemanticDescriptionGenerator:
    """Generates rich semantic descriptions for RAG retrieval."""

    def generate(self, obj: Dict[str, Any]) -> str:
        """
        Generate a semantic description from object metadata.

        The description should be rich enough for embedding-based retrieval.
        """
        parts = []

        # Category and name
        category = obj.get("category", "object")
        name = obj.get("name", obj.get("id", ""))
        parts.append(f"A {category}")
        if name and name != category:
            parts.append(f"named {name}")

        # Description if available
        description = obj.get("description", "")
        if description:
            parts.append(f"- {description}")

        # Size information
        dimensions = obj.get("dimensions_est", {})
        if isinstance(dimensions, dict) and any(dimensions.values()):
            w = dimensions.get("width", 0)
            d = dimensions.get("depth", 0)
            h = dimensions.get("height", 0)
            if w > 0 and d > 0 and h > 0:
                parts.append(f"approximately {w:.2f}m x {d:.2f}m x {h:.2f}m")

        # Material
        physics_hints = obj.get("physics_hints", {})
        material_type = physics_hints.get("material_type", "")
        if material_type:
            parts.append(f"made of {material_type}")

        # Affordances
        semantics = obj.get("semantics", {})
        affordances = semantics.get("affordances", [])
        if affordances:
            aff_names = []
            for aff in affordances:
                if isinstance(aff, str):
                    aff_names.append(aff.lower())
                elif isinstance(aff, dict):
                    aff_names.append(aff.get("type", "").lower())
            if aff_names:
                parts.append(f"can be {', '.join(aff_names)}")

        # Sim role context
        sim_role = obj.get("sim_role", "")
        if sim_role == "articulated_furniture":
            parts.append("with moving parts")
        elif sim_role == "articulated_appliance":
            parts.append("an interactive appliance")
        elif sim_role == "manipulable_object":
            parts.append("suitable for robot manipulation")

        return " ".join(parts)


# =============================================================================
# Asset Index Builder
# =============================================================================


class AssetIndexBuilder:
    """
    Builds Genie Sim asset index from BlueprintPipeline scene manifest.

    Usage:
        builder = AssetIndexBuilder()
        index = builder.build(manifest_dict)
        index.save(Path("output/asset_index.json"))
    """

    def __init__(
        self,
        generate_embeddings: bool = False,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize asset index builder.

        Args:
            generate_embeddings: Whether to generate embeddings (requires model)
            embedding_model: Embedding model name (e.g., "qwen-text-embedding-v4")
            embedding_provider: Embedding provider ("openai" or "qwen")
            verbose: Print progress
        """
        self.verbose = verbose
        self.generate_embeddings = generate_embeddings
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.description_generator = SemanticDescriptionGenerator()
        self.environment = os.getenv("GENIESIM_ENV", os.getenv("BP_ENV", "development")).lower()

        # Embedding client (initialized lazily)
        self._embedding_client = None
        self._embedding_config: Optional[Dict[str, str]] = None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[ASSET-INDEX-BUILDER] [{level}] {msg}")

    def build(
        self,
        manifest: Dict[str, Any],
        usd_base_path: Optional[str] = None,
    ) -> GenieSimAssetIndex:
        """
        Build Genie Sim asset index from BlueprintPipeline manifest.

        Args:
            manifest: BlueprintPipeline scene_manifest.json as dict
            usd_base_path: Base path for USD files

        Returns:
            GenieSimAssetIndex ready for Genie Sim
        """
        self.log("Building Genie Sim asset index")

        objects = manifest.get("objects", [])
        scene_id = manifest.get("scene_id", "unknown")

        assets = []
        for obj in objects:
            asset = self._build_asset(obj, usd_base_path)
            if asset:
                assets.append(asset)

        self.log(f"Built {len(assets)} assets from {len(objects)} objects")

        # Generate embeddings if requested
        if self.generate_embeddings and assets:
            self._generate_embeddings(assets)

        # Build metadata
        metadata = {
            "source_scene": scene_id,
            "total_assets": len(assets),
            "commercial_assets": sum(1 for a in assets if a.commercial_ok),
            "embedding_model": self.embedding_model if self.generate_embeddings else None,
        }

        return GenieSimAssetIndex(assets=assets, metadata=metadata)

    def _is_production(self) -> bool:
        return self.environment == "production"

    def _resolve_embedding_config(self) -> Optional[Dict[str, str]]:
        if self._embedding_config is not None:
            return self._embedding_config

        provider = self.embedding_provider
        if provider is None:
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"):
                provider = "qwen"

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = self.embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        elif provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
            base_url = os.getenv(
                "QWEN_BASE_URL",
                os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
            model = self.embedding_model or os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")
        else:
            self._embedding_config = None
            return None

        if not api_key:
            self._embedding_config = None
            return None

        self.embedding_provider = provider
        self.embedding_model = model
        self._embedding_config = {
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url.rstrip("/"),
            "model": model,
        }
        return self._embedding_config

    def _build_asset(
        self,
        obj: Dict[str, Any],
        usd_base_path: Optional[str],
    ) -> Optional[GenieSimAsset]:
        """Build a single asset entry from object data."""
        try:
            obj_id = str(obj.get("id", ""))
            if not obj_id:
                return None

            # Skip background/shell objects
            sim_role = obj.get("sim_role", "unknown")
            if sim_role in ["background", "scene_shell"]:
                return None

            # Get USD path
            asset_data = obj.get("asset", {})
            usd_path = asset_data.get("path", "")
            if usd_base_path and usd_path and not usd_path.startswith("/"):
                usd_path = f"{usd_base_path}/{usd_path}"

            # Generate semantic description
            semantic_description = self.description_generator.generate(obj)

            # Get categories
            category = (obj.get("category") or "object").lower()
            categories = CATEGORY_MAPPING.get(category, CATEGORY_MAPPING["object"])

            # Get physics properties
            physics = obj.get("physics", {})
            physics_hints = obj.get("physics_hints", {})

            mass = physics.get("mass")
            if mass is None:
                # Estimate mass from dimensions and material
                dimensions = obj.get("dimensions_est", {})
                if isinstance(dimensions, dict):
                    volume = (
                        dimensions.get("width", 0.1) *
                        dimensions.get("depth", 0.1) *
                        dimensions.get("height", 0.1)
                    )
                    material_type = physics_hints.get("material_type", "default")
                    density = MATERIAL_PHYSICS.get(material_type, MATERIAL_PHYSICS["default"]).density
                    mass = volume * density

            # Get material properties
            material_type = physics_hints.get("material_type", "default")
            material = MATERIAL_PHYSICS.get(material_type, MATERIAL_PHYSICS["default"])

            # Override with explicit values if present
            if "friction" in physics:
                material = GenieSimMaterial(
                    friction=physics["friction"],
                    restitution=material.restitution,
                    density=material.density,
                )
            if "restitution" in physics:
                material = GenieSimMaterial(
                    friction=material.friction,
                    restitution=physics["restitution"],
                    density=material.density,
                )

            # Get texture variants
            texture_variants = asset_data.get("variants", {})

            # Collision hull (assume same as mesh or explicit)
            collision_hull_path = physics.get("collision_shape")
            if not collision_hull_path and usd_path:
                # Assume collision hull is derived from mesh
                collision_hull_path = usd_path.replace(".usdz", "_collision.usda")

            # Determine commercial status
            # Assets from BlueprintPipeline are assumed commercial OK
            # unless explicitly marked otherwise
            asset_source = asset_data.get("source", "blueprintpipeline")
            commercial_ok = asset_source not in ["geniesim_assets", "external_nc"]

            # Preserve BP metadata
            bp_metadata = {
                "sim_role": sim_role,
                "category": category,
                "affordances": obj.get("semantics", {}).get("affordances", []),
                "articulation": obj.get("articulation", {}),
            }

            return GenieSimAsset(
                asset_id=obj_id,
                usd_path=usd_path,
                semantic_description=semantic_description,
                categories=list(categories),
                mass=mass,
                material=material,
                collision_hull_path=collision_hull_path,
                texture_variants=texture_variants,
                commercial_ok=commercial_ok,
                license="proprietary" if commercial_ok else "unknown",
                source=asset_source,
                bp_metadata=bp_metadata,
            )

        except Exception as e:
            self.log(f"Failed to build asset for {obj.get('id', 'unknown')}: {e}", "WARNING")
            return None

    def _generate_embeddings(self, assets: List[GenieSimAsset]) -> None:
        """Generate embeddings for all assets."""
        self.log(f"Generating embeddings for {len(assets)} assets...")
        production_mode = self._is_production()

        try:
            for asset in assets:
                asset.embedding = self._get_embedding(asset.semantic_description)

            self.log(f"Generated embeddings for {len(assets)} assets")

        except Exception as e:
            if production_mode:
                raise RuntimeError(f"Embedding generation failed in production: {e}") from e
            self.log(f"Embedding generation failed: {e}", "WARNING")

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.

        Uses a configured embedding provider:
        - OpenAI embeddings via OPENAI_API_KEY
        - Qwen embeddings via QWEN_API_KEY / DASHSCOPE_API_KEY
        """
        config = self._resolve_embedding_config()
        if not config:
            if self._is_production():
                raise RuntimeError(
                    "Embeddings are required in production but no embedding provider "
                    "is configured. Set OPENAI_API_KEY or QWEN_API_KEY/DASHSCOPE_API_KEY."
                )
            self.log("Embedding provider unavailable; using placeholder embeddings.", "WARNING")
            return self._get_placeholder_embedding(text)

        try:
            return self._request_embedding(text, config)
        except Exception as e:
            if self._is_production():
                raise
            self.log(f"Embedding request failed; using placeholder embeddings: {e}", "WARNING")
            return self._get_placeholder_embedding(text)

    def _request_embedding(self, text: str, config: Dict[str, str]) -> List[float]:
        payload = {
            "model": config["model"],
            "input": text,
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{config['base_url']}/embeddings"
        request = url_request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}",
            },
            method="POST",
        )

        try:
            with url_request.urlopen(request, timeout=30) as response:
                response_body = response.read()
        except url_error.HTTPError as e:
            raise RuntimeError(f"Embedding request failed: {e.code} {e.reason}") from e
        except url_error.URLError as e:
            raise RuntimeError(f"Embedding request failed: {e.reason}") from e

        data = json.loads(response_body.decode("utf-8"))
        if "data" not in data or not data["data"]:
            raise RuntimeError("Embedding response missing data")

        embedding = data["data"][0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Embedding response missing embedding vector")

        return embedding

    def _get_placeholder_embedding(self, text: str) -> List[float]:
        """Generate a deterministic placeholder embedding (dev only)."""
        import hashlib

        text_hash = hashlib.sha256(text.encode()).digest()

        embedding = []
        for i in range(0, min(len(text_hash), 32), 4):
            val = int.from_bytes(text_hash[i:i + 4], "big")
            embedding.append((val / (2**32)) * 2 - 1)

        while len(embedding) < 2048:
            embedding.extend(embedding[:min(len(embedding), 2048 - len(embedding))])

        return embedding[:2048]


# =============================================================================
# Convenience Functions
# =============================================================================


def build_asset_index(
    manifest_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> GenieSimAssetIndex:
    """
    Convenience function to build asset index from manifest file.

    Args:
        manifest_path: Path to scene_manifest.json
        output_path: Optional path to save asset_index.json
        verbose: Print progress

    Returns:
        GenieSimAssetIndex
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    builder = AssetIndexBuilder(verbose=verbose)
    index = builder.build(manifest)

    if output_path:
        index.save(output_path)

    return index
