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
    - Embeddings sized to configured embedding_dim (optional, requires embedding model)

References:
    - Genie Sim 3.0 Paper: https://arxiv.org/html/2601.02078v1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from urllib import error as url_error
from urllib import request as url_request
import warnings

import numpy as np

from tools.config.production_mode import resolve_pipeline_environment, resolve_production_mode
from tools.quality_reports import COMMERCIAL_OK_LICENSES, LicenseType
from tools.validation import ALLOWED_ASSET_CATEGORIES, ValidationError, validate_category


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

    # Embedding (size from embedding_dim, generated separately)
    embedding: Optional[List[float]] = None
    embedding_status: str = "pending"

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
            "embedding_status": self.embedding_status,
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
            "schema_version": "3.0",
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
    "steel": GenieSimMaterial(friction=0.45, restitution=0.2, density=7850.0),
    "stainless_steel": GenieSimMaterial(friction=0.4, restitution=0.2, density=8000.0),
    "cast_iron": GenieSimMaterial(friction=0.5, restitution=0.15, density=7200.0),
    "aluminum": GenieSimMaterial(friction=0.4, restitution=0.25, density=2700.0),
    "plastic": GenieSimMaterial(friction=0.5, restitution=0.3, density=1200.0),
    "wood": GenieSimMaterial(friction=0.6, restitution=0.15, density=700.0),
    "glass": GenieSimMaterial(friction=0.3, restitution=0.1, density=2500.0),
    "ceramic": GenieSimMaterial(friction=0.5, restitution=0.1, density=2300.0),
    "stone": GenieSimMaterial(friction=0.6, restitution=0.05, density=2600.0),
    "concrete": GenieSimMaterial(friction=0.7, restitution=0.05, density=2400.0),
    "brick": GenieSimMaterial(friction=0.7, restitution=0.05, density=1800.0),
    "leather": GenieSimMaterial(friction=0.6, restitution=0.1, density=860.0),
    "fabric": GenieSimMaterial(friction=0.7, restitution=0.05, density=500.0),
    "foam": GenieSimMaterial(friction=0.8, restitution=0.3, density=50.0),
    "rubber": GenieSimMaterial(friction=0.9, restitution=0.6, density=1100.0),
    "paper": GenieSimMaterial(friction=0.5, restitution=0.05, density=700.0),
    "cardboard": GenieSimMaterial(friction=0.5, restitution=0.1, density=300.0),
    "water": GenieSimMaterial(friction=0.05, restitution=0.0, density=1000.0),
    "default": GenieSimMaterial(friction=0.5, restitution=0.1, density=1000.0),
}

MATERIAL_ALIASES = {
    "metal": "steel",
    "stainless": "stainless_steel",
    "stainlesssteel": "stainless_steel",
    "aluminium": "aluminum",
    "aluminium_alloy": "aluminum",
    "iron": "cast_iron",
    "granite": "stone",
    "marble": "stone",
    "rock": "stone",
    "cement": "concrete",
    "brickwork": "brick",
    "timber": "wood",
    "plywood": "wood",
    "cloth": "fabric",
    "textile": "fabric",
    "leatherette": "leather",
    "styrofoam": "foam",
    "foam_rubber": "foam",
    "rubber_foam": "foam",
    "porcelain": "ceramic",
    "liquid": "water",
}

MATERIAL_PHYSICS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "policy_configs" / "material_physics.json"
)


def normalize_material_type(material_type: Optional[str]) -> str:
    if not material_type:
        return "default"
    normalized = str(material_type).strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    return MATERIAL_ALIASES.get(normalized, normalized)


def _handle_missing_material_physics_config(path: Path) -> None:
    message = (
        "Material physics config not found. Expected file at "
        f"{path}."
    )
    if resolve_production_mode():
        raise FileNotFoundError(message)
    warnings.warn(message)


def _load_material_physics_overrides(
    config_path: Optional[Path] = None,
) -> Dict[str, GenieSimMaterial]:
    path = config_path or MATERIAL_PHYSICS_CONFIG_PATH
    if not path.exists():
        _handle_missing_material_physics_config(path)
        return {}

    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Failed to read material physics overrides from {path}: {exc}")
        return {}

    if not isinstance(payload, dict):
        warnings.warn(f"Material physics overrides in {path} must be a JSON object.")
        return {}

    materials = payload.get("materials")
    if materials is None:
        warnings.warn(f"Material physics overrides in {path} missing 'materials' mapping.")
        return {}
    if not isinstance(materials, dict):
        warnings.warn(f"Material physics overrides in {path} must define a 'materials' object.")
        return {}

    overrides: Dict[str, GenieSimMaterial] = {}
    for name, values in materials.items():
        if not isinstance(name, str) or not name.strip():
            warnings.warn(
                f"Ignoring material physics override with invalid name: {name!r}."
            )
            continue
        if not isinstance(values, dict):
            warnings.warn(
                f"Ignoring material physics override for {name!r}: expected an object."
            )
            continue

        normalized_name = normalize_material_type(name)
        base_material = MATERIAL_PHYSICS.get(normalized_name, MATERIAL_PHYSICS["default"])

        invalid = False
        friction = base_material.friction
        restitution = base_material.restitution
        density = base_material.density

        for key, setter in (
            ("friction", "friction"),
            ("restitution", "restitution"),
            ("density", "density"),
        ):
            if key not in values:
                continue
            value = values[key]
            if not isinstance(value, (int, float)):
                warnings.warn(
                    f"Ignoring material physics override for {name!r}: "
                    f"{key} must be numeric."
                )
                invalid = True
                break
            if value < 0 or (key == "density" and value == 0):
                warnings.warn(
                    f"Ignoring material physics override for {name!r}: "
                    f"{key} must be positive."
                )
                invalid = True
                break
            if setter == "friction":
                friction = float(value)
            elif setter == "restitution":
                restitution = float(value)
            else:
                density = float(value)

        if invalid:
            continue

        overrides[normalized_name] = GenieSimMaterial(
            friction=friction,
            restitution=restitution,
            density=density,
        )

    return overrides


_MATERIAL_OVERRIDES = _load_material_physics_overrides()
if _MATERIAL_OVERRIDES:
    MATERIAL_PHYSICS = {**MATERIAL_PHYSICS, **_MATERIAL_OVERRIDES}


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
        material_type = normalize_material_type(physics_hints.get("material_type", ""))
        if material_type != "default":
            parts.append(f"made of {material_type.replace('_', ' ')}")

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
        embedding_dim: Optional[int] = None,
        verbose: bool = True,
        strict_category_validation: Optional[bool] = None,
        require_embeddings: Optional[bool] = None,
        allow_embedding_fallback: Optional[bool] = None,
    ):
        """
        Initialize asset index builder.

        Args:
            generate_embeddings: Whether to generate embeddings (requires model)
            embedding_model: Embedding model name (e.g., "qwen-text-embedding-v4")
            embedding_provider: Embedding provider ("openai" or "qwen")
            embedding_dim: Embedding dimensionality (overrides GENIESIM_EMBEDDING_DIM)
            require_embeddings: Require real embeddings (disallow placeholders)
            allow_embedding_fallback: Allow placeholder embeddings in production
            verbose: Print progress
        """
        default_embedding_dims = {768, 1024, 2048, 3072}
        allowed_embedding_dims_env = os.getenv("GENIESIM_ALLOWED_EMBEDDING_DIMS", "").strip()
        if allowed_embedding_dims_env:
            raw_tokens = [
                token.strip()
                for token in re.split(r"[,\s]+", allowed_embedding_dims_env)
                if token.strip()
            ]
            if not raw_tokens:
                raise ValueError(
                    "GENIESIM_ALLOWED_EMBEDDING_DIMS must be a comma-separated list of "
                    "positive integers; got an empty value."
                )
            parsed_dims: List[int] = []
            for token in raw_tokens:
                try:
                    dim = int(token)
                except ValueError as exc:
                    raise ValueError(
                        "GENIESIM_ALLOWED_EMBEDDING_DIMS must be a comma-separated list of "
                        f"positive integers; got {allowed_embedding_dims_env!r}."
                    ) from exc
                if dim <= 0:
                    raise ValueError(
                        "GENIESIM_ALLOWED_EMBEDDING_DIMS must contain only positive integers; "
                        f"got {allowed_embedding_dims_env!r}."
                    )
                parsed_dims.append(dim)
            allowed_embedding_dims = set(parsed_dims)
        else:
            allowed_embedding_dims = default_embedding_dims
        self.verbose = verbose
        self.generate_embeddings = generate_embeddings
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.description_generator = SemanticDescriptionGenerator()
        self.environment = resolve_pipeline_environment()
        if strict_category_validation is None:
            strict_env = os.getenv("GENIESIM_STRICT_CATEGORY", "")
            self.strict_category_validation = strict_env.lower() in {"1", "true", "yes", "on"}
        else:
            self.strict_category_validation = strict_category_validation
        if require_embeddings is None:
            require_env = os.getenv("REQUIRE_EMBEDDINGS")
            if require_env is None:
                require_env = os.getenv("DISALLOW_PLACEHOLDER_EMBEDDINGS")
            if require_env is not None:
                self.require_embeddings = require_env.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                self.require_embeddings = self._is_production()
        else:
            self.require_embeddings = require_embeddings
        if allow_embedding_fallback is None:
            fallback_env = os.getenv("ALLOW_EMBEDDING_FALLBACK")
            if fallback_env is not None:
                self.allow_embedding_fallback = fallback_env.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                self.allow_embedding_fallback = False
        else:
            self.allow_embedding_fallback = allow_embedding_fallback
        local_fallback_env = os.getenv("LOCAL_EMBEDDING_FALLBACK")
        if local_fallback_env is not None:
            self.local_embedding_fallback = local_fallback_env.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self.local_embedding_fallback = False
        self.local_embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        )

        if embedding_dim is None:
            embedding_dim_env = os.getenv("GENIESIM_EMBEDDING_DIM", "").strip()
            if embedding_dim_env:
                try:
                    embedding_dim = int(embedding_dim_env)
                except ValueError as exc:
                    raise ValueError(
                        "GENIESIM_EMBEDDING_DIM must be an integer value, "
                        f"got {embedding_dim_env!r}."
                    ) from exc
        if embedding_dim is None:
            embedding_dim = 2048
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if embedding_dim not in allowed_embedding_dims:
            raise ValueError(
                "Unsupported embedding_dim. Genie Sim requires one of "
                f"{sorted(allowed_embedding_dims)} (configured via GENIESIM_ALLOWED_EMBEDDING_DIMS); "
                f"got {embedding_dim}."
            )
        self.embedding_dim = embedding_dim

        # Embedding client (initialized lazily)
        self._embedding_client = None
        self._embedding_config: Optional[Dict[str, str]] = None
        self._local_embedding_client = None

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

        embedding_errors: List[Dict[str, str]] = []
        embedding_failure_rate = 0.0
        embedding_failures = 0
        placeholder_embeddings = 0
        local_fallback_embeddings = 0

        # Generate embeddings if requested
        if self.generate_embeddings and assets:
            embedding_result = self._generate_embeddings(assets)
            embedding_errors = embedding_result["errors"]
            embedding_failure_rate = embedding_result["failure_rate"]
            embedding_failures = embedding_result["failure_count"]
            placeholder_embeddings = embedding_result["placeholder_count"]
            local_fallback_embeddings = embedding_result["local_fallback_count"]
        else:
            for asset in assets:
                asset.embedding_status = "not_requested"

        if not self.generate_embeddings:
            embedding_mode = "disabled"
        elif placeholder_embeddings > 0 and self._fallback_enabled():
            embedding_mode = "fallback"
        elif local_fallback_embeddings > 0:
            embedding_mode = "fallback_local"
        elif self._resolve_embedding_config() is not None:
            embedding_mode = "provider"
        else:
            embedding_mode = "placeholder"

        # Build metadata
        metadata = {
            "source_scene": scene_id,
            "total_assets": len(assets),
            "commercial_assets": sum(1 for a in assets if a.commercial_ok),
            "embedding_model": self.embedding_model if self.generate_embeddings else None,
            "embedding_dim": self.embedding_dim,
            "embedding_errors": embedding_errors,
            "embedding_failure_rate": embedding_failure_rate,
            "embedding_failures": embedding_failures,
            "placeholder_embeddings_allowed": not self.require_embeddings,
            "placeholder_embeddings_used": placeholder_embeddings,
            "local_fallback_embeddings_used": local_fallback_embeddings,
            "embedding_mode": embedding_mode,
            "embedding_fallback_allowed": self.allow_embedding_fallback,
        }

        if embedding_failures > 0 and not self._is_production():
            metadata["warning_banner"] = (
                f"Embedding generation had {embedding_failures} failures; placeholder "
                "embeddings were used for affected assets."
            )

        return GenieSimAssetIndex(assets=assets, metadata=metadata)

    def _is_production(self) -> bool:
        return self.environment == "production"

    def _fallback_enabled(self) -> bool:
        return self._is_production() and self.allow_embedding_fallback and not self.require_embeddings

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
            category_raw = obj.get("category") or "object"
            category = validate_category(
                category_raw,
                allowed_categories=ALLOWED_ASSET_CATEGORIES,
                strict=False,
            )
            if category not in CATEGORY_MAPPING:
                self.log(
                    f"Unknown category '{category_raw}' for asset {obj_id}. Using default categories.",
                    "WARNING",
                )
                if self.strict_category_validation:
                    raise ValidationError(
                        f"Unknown category '{category_raw}'",
                        field="category",
                        value=category_raw,
                    )
                categories = CATEGORY_MAPPING["object"]
            else:
                categories = CATEGORY_MAPPING[category]

            # Get physics properties
            physics = obj.get("physics", {})
            physics_hints = obj.get("physics_hints", {})
            raw_material_type = physics_hints.get("material_type", "")
            material_type = normalize_material_type(raw_material_type)
            if raw_material_type and material_type not in MATERIAL_PHYSICS:
                self.log(
                    f"Unknown material_type '{raw_material_type}' for asset {obj_id}; using default.",
                    "WARNING",
                )
                material_type = "default"

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
                    density = MATERIAL_PHYSICS.get(material_type, MATERIAL_PHYSICS["default"]).density
                    mass = volume * density

            # Get material properties
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

            asset_source = asset_data.get("source", "blueprintpipeline")
            commercial_ok, license_value = self._resolve_commercial_status(
                obj=obj,
                asset_data=asset_data,
                asset_source=asset_source,
            )

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
                license=license_value,
                source=asset_source,
                bp_metadata=bp_metadata,
            )

        except ValidationError:
            if self.strict_category_validation:
                raise
            self.log(f"Failed to build asset for {obj.get('id', 'unknown')}: invalid input", "WARNING")
            return None
        except Exception as e:
            self.log(f"Failed to build asset for {obj.get('id', 'unknown')}: {e}", "WARNING")
            return None

    @staticmethod
    def _normalize_license_string(value: str) -> str:
        normalized = str(value).strip().lower()
        normalized = normalized.replace("_", "-")
        normalized = normalized.replace("(", " ").replace(")", " ")
        normalized = " ".join(normalized.split())
        for prefix in ("license:", "spdx:"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        return normalized

    @staticmethod
    def _parse_license_value(value: Optional[str]) -> tuple[Optional[LicenseType], Dict[str, Any]]:
        if not value:
            return None, {"raw": value, "normalized": None, "match": None, "candidates": []}
        raw = str(value).strip()
        normalized = GenieSimAssetIndexBuilder._normalize_license_string(raw)
        license_map = {
            "cc0": LicenseType.CC0,
            "cc0-1.0": LicenseType.CC0_1_0,
            "cc-by": LicenseType.CC_BY,
            "cc-by-4.0": LicenseType.CC_BY_4_0,
            "cc-by-sa": LicenseType.CC_BY_SA,
            "cc-by-nc": LicenseType.CC_BY_NC,
            "cc-by-nc-sa": LicenseType.CC_BY_NC_SA,
            "academic-only": LicenseType.ACADEMIC_ONLY,
            "research-only": LicenseType.RESEARCH_ONLY,
            "mit": LicenseType.MIT,
            "apache-2.0": LicenseType.APACHE_2,
            "bsd-2-clause": LicenseType.BSD_2_CLAUSE,
            "bsd-3-clause": LicenseType.BSD_3_CLAUSE,
            "gpl-3.0": LicenseType.GPL_3_0,
            "lgpl-3.0": LicenseType.LGPL_3_0,
            "mpl-2.0": LicenseType.MPL_2_0,
            "proprietary-commercial": LicenseType.PROPRIETARY_COMMERCIAL,
            "proprietary": LicenseType.PROPRIETARY_COMMERCIAL,
            "nvidia-omniverse": LicenseType.NVIDIA_OMNIVERSE,
            "simready": LicenseType.SIMREADY,
        }
        synonym_map = {
            "cc by": LicenseType.CC_BY,
            "cc-by 4.0": LicenseType.CC_BY_4_0,
            "cc by 4.0": LicenseType.CC_BY_4_0,
            "creative commons attribution 4.0": LicenseType.CC_BY_4_0,
            "creative commons attribution 4.0 international": LicenseType.CC_BY_4_0,
            "cc0 1.0": LicenseType.CC0_1_0,
            "cc0-1": LicenseType.CC0_1_0,
            "creative commons zero 1.0": LicenseType.CC0_1_0,
            "apache": LicenseType.APACHE_2,
            "apache-2": LicenseType.APACHE_2,
            "apache 2": LicenseType.APACHE_2,
            "apache 2.0": LicenseType.APACHE_2,
            "apache license 2.0": LicenseType.APACHE_2,
            "mit license": LicenseType.MIT,
            "bsd 2 clause": LicenseType.BSD_2_CLAUSE,
            "bsd 2-clause": LicenseType.BSD_2_CLAUSE,
            "bsd 3 clause": LicenseType.BSD_3_CLAUSE,
            "bsd 3-clause": LicenseType.BSD_3_CLAUSE,
            "gpl-3": LicenseType.GPL_3_0,
            "gpl-3.0-only": LicenseType.GPL_3_0,
            "gpl v3": LicenseType.GPL_3_0,
            "gplv3": LicenseType.GPL_3_0,
            "lgpl-3": LicenseType.LGPL_3_0,
            "lgpl-3.0-only": LicenseType.LGPL_3_0,
            "lgpl v3": LicenseType.LGPL_3_0,
            "lgplv3": LicenseType.LGPL_3_0,
            "mpl 2.0": LicenseType.MPL_2_0,
            "mozilla public license 2.0": LicenseType.MPL_2_0,
            "nvidia": LicenseType.NVIDIA_OMNIVERSE,
        }

        if normalized in license_map:
            return license_map[normalized], {
                "raw": raw,
                "normalized": normalized,
                "match": "exact",
                "candidates": [],
            }
        if normalized in synonym_map:
            return synonym_map[normalized], {
                "raw": raw,
                "normalized": normalized,
                "match": "synonym",
                "candidates": [],
            }

        if re.search(r"\b(or|and)\b|/|,", normalized):
            parts = re.split(r"\s*(?:/|,|\bor\b|\band\b)\s*", normalized)
            matches: List[LicenseType] = []
            for part in parts:
                if not part:
                    continue
                part = part.strip()
                if part in license_map:
                    matches.append(license_map[part])
                elif part in synonym_map:
                    matches.append(synonym_map[part])
            unique_matches = list(dict.fromkeys(matches))
            if len(unique_matches) > 1:
                return None, {
                    "raw": raw,
                    "normalized": normalized,
                    "match": "ambiguous",
                    "candidates": [m.value for m in unique_matches],
                }

        return None, {
            "raw": raw,
            "normalized": normalized,
            "match": "unknown",
            "candidates": [],
        }

    def _resolve_commercial_status(
        self,
        *,
        obj: Dict[str, Any],
        asset_data: Dict[str, Any],
        asset_source: str,
    ) -> tuple[bool, str]:
        license_raw = asset_data.get("license") or obj.get("license")
        provenance = asset_data.get("provenance") or obj.get("provenance")
        if not license_raw and isinstance(provenance, dict):
            provenance_license = provenance.get("license")
            if isinstance(provenance_license, dict):
                license_raw = provenance_license.get("type") or license_raw
            elif isinstance(provenance_license, str):
                license_raw = provenance_license

        license_type, parse_info = self._parse_license_value(license_raw)
        commercial_ok: Optional[bool] = None
        if license_type is not None:
            commercial_ok = license_type in COMMERCIAL_OK_LICENSES

        provenance_license = None
        if isinstance(provenance, dict):
            provenance_license = provenance.get("license")

        if license_type is None and license_raw:
            if isinstance(provenance_license, dict) and "commercial_ok" in provenance_license:
                commercial_ok = bool(provenance_license.get("commercial_ok"))
            else:
                commercial_ok = False
        else:
            if commercial_ok is None and isinstance(provenance_license, dict):
                if "commercial_ok" in provenance_license:
                    commercial_ok = bool(provenance_license.get("commercial_ok"))
            if commercial_ok is None and isinstance(provenance, dict) and "commercial_ok" in provenance:
                commercial_ok = bool(provenance.get("commercial_ok"))
            if commercial_ok is None and isinstance(asset_data.get("commercial_ok"), bool):
                commercial_ok = bool(asset_data.get("commercial_ok"))
            if commercial_ok is None:
                commercial_ok = asset_source not in ["geniesim_assets", "external_nc"]

        if license_type is not None:
            license_value = license_type.value
        elif license_raw:
            license_value = "unknown"
        else:
            license_value = "proprietary" if commercial_ok else "unknown"

        if license_raw and license_type is None:
            log_payload = {
                "event": "license_resolution",
                "status": parse_info.get("match"),
                "asset_id": obj.get("id"),
                "license_raw": license_raw,
                "normalized": parse_info.get("normalized"),
                "candidates": parse_info.get("candidates", []),
                "commercial_ok": commercial_ok,
                "provenance_commercial_ok": (
                    provenance_license.get("commercial_ok")
                    if isinstance(provenance_license, dict)
                    else None
                ),
            }
            self.log(json.dumps(log_payload), "WARNING")

        return commercial_ok, license_value

    def _generate_embeddings(self, assets: List[GenieSimAsset]) -> Dict[str, Any]:
        """Generate embeddings for all assets."""
        self.log(f"Generating embeddings for {len(assets)} assets...")
        production_mode = self._is_production()
        fallback_enabled = self._fallback_enabled()
        local_fallback_enabled = self._local_fallback_enabled()
        errors: List[Dict[str, str]] = []
        placeholder_count = 0
        local_fallback_count = 0

        for asset in assets:
            try:
                embedding, status, error_message = self._get_embedding_with_status(
                    asset.semantic_description
                )
            except Exception as e:
                if self.require_embeddings:
                    raise RuntimeError(
                        f"Embedding generation failed with REQUIRE_EMBEDDINGS enabled for asset "
                        f"{asset.asset_id}: {e}"
                    ) from e
                if production_mode and not fallback_enabled:
                    raise RuntimeError(
                        f"Embedding generation failed in production for asset {asset.asset_id}: {e}"
                    ) from e
                embedding = self._get_placeholder_embedding(asset.semantic_description)
                status = "fallback" if fallback_enabled else "placeholder"
                error_message = str(e)

            asset.embedding = embedding
            asset.embedding_status = status
            if status in {"placeholder", "fallback"}:
                placeholder_count += 1
            elif status == "fallback_local":
                local_fallback_count += 1

            if status not in {"ok", "fallback_local"}:
                errors.append({"asset_id": asset.asset_id, "error": error_message})
                if production_mode and self.require_embeddings:
                    raise RuntimeError(
                        f"Embedding generation failed in production for asset {asset.asset_id}: "
                        f"{error_message}"
                    )

        if production_mode and placeholder_count > 0 and not fallback_enabled:
            raise RuntimeError(
                "Placeholder embeddings are not allowed in production. "
                "Configure an embedding provider by setting OPENAI_API_KEY or "
                "QWEN_API_KEY/DASHSCOPE_API_KEY."
            )
        if production_mode and placeholder_count > 0 and fallback_enabled:
            self.log(
                "Placeholder embeddings used in production because "
                "ALLOW_EMBEDDING_FALLBACK is enabled.",
                "WARNING",
            )
        if production_mode and local_fallback_count > 0 and local_fallback_enabled:
            self.log(
                "Local fallback embeddings used in production because "
                "LOCAL_EMBEDDING_FALLBACK is enabled.",
                "WARNING",
            )

        failure_count = len(errors)
        failure_rate = failure_count / len(assets) if assets else 0.0
        self.log(
            f"Embedding failure rate: {failure_rate:.2%} ({failure_count}/{len(assets)})"
        )
        self.log(
            "Placeholder embeddings allowed: "
            f"{not self.require_embeddings}; used: {placeholder_count}"
        )
        if failure_count > 0:
            self.log(
                f"Embedding generation had {failure_count} failures", "WARNING"
            )
        else:
            self.log(f"Generated embeddings for {len(assets)} assets")

        return {
            "errors": errors,
            "failure_rate": failure_rate,
            "failure_count": failure_count,
            "placeholder_count": placeholder_count,
            "local_fallback_count": local_fallback_count,
        }

    def _local_fallback_enabled(self) -> bool:
        return self.local_embedding_fallback

    def _get_embedding_with_status(self, text: str) -> tuple[List[float], str, str]:
        config = self._resolve_embedding_config()
        if not config:
            if self._local_fallback_enabled():
                return self._get_local_embedding_with_status(text)
            if self._is_production() and not self._fallback_enabled():
                raise RuntimeError(
                    "Embeddings are required in production but no embedding provider is configured. "
                    "Set OPENAI_API_KEY or QWEN_API_KEY/DASHSCOPE_API_KEY."
                )
            if self.require_embeddings:
                raise RuntimeError(
                    "Embeddings are required but no embedding provider is configured. "
                    "Set OPENAI_API_KEY or QWEN_API_KEY/DASHSCOPE_API_KEY."
                )
            fallback_status = "fallback" if self._fallback_enabled() else "placeholder"
            self.log("Embedding provider unavailable; using placeholder embeddings.", "WARNING")
            return (
                self._get_placeholder_embedding(text),
                fallback_status,
                "Embedding provider unavailable; placeholder embedding used.",
            )

        try:
            embedding = self._request_embedding(text, config)
            embedding = self._normalize_embedding_length(embedding, source="provider")
            return embedding, "ok", ""
        except Exception as e:
            if self._local_fallback_enabled():
                return self._get_local_embedding_with_status(
                    text,
                    error_prefix=f"Embedding request failed; using local fallback: {e}",
                )
            if self.require_embeddings:
                raise RuntimeError(
                    "Embedding request failed while REQUIRE_EMBEDDINGS is enabled; "
                    f"placeholder embeddings are disallowed. Error: {e}"
                ) from e
            if self._is_production() and not self._fallback_enabled():
                raise
            fallback_status = "fallback" if self._fallback_enabled() else "placeholder"
            self.log(f"Embedding request failed; using placeholder embeddings: {e}", "WARNING")
            return (
                self._get_placeholder_embedding(text),
                fallback_status,
                f"Embedding request failed; placeholder embedding used: {e}",
            )

    def _get_local_embedding_with_status(
        self,
        text: str,
        error_prefix: Optional[str] = None,
    ) -> tuple[List[float], str, str]:
        try:
            embedding = self._get_local_embedding(text)
            embedding = self._normalize_embedding_length(
                embedding,
                source="local_fallback",
                allow_resize_in_production=True,
            )
            message = error_prefix or "Embedding provider unavailable; using local fallback embeddings."
            self.log(message, "WARNING")
            return embedding, "fallback_local", message
        except Exception as e:
            if self.require_embeddings:
                raise RuntimeError(
                    "Local embedding fallback failed while REQUIRE_EMBEDDINGS is enabled; "
                    f"placeholders are disallowed. Error: {e}"
                ) from e
            if self._is_production() and not self._fallback_enabled():
                raise
            fallback_status = "fallback" if self._fallback_enabled() else "placeholder"
            self.log(f"Local embedding fallback failed; using placeholder embeddings: {e}", "WARNING")
            return (
                self._get_placeholder_embedding(text),
                fallback_status,
                f"Local embedding fallback failed; placeholder embedding used: {e}",
            )

    def _get_local_embedding(self, text: str) -> List[float]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embedding fallback requested but sentence-transformers is not installed."
            ) from exc

        if self._local_embedding_client is None:
            self._local_embedding_client = SentenceTransformer(self.local_embedding_model)

        embedding = self._local_embedding_client.encode([text])
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        if not embedding or not isinstance(embedding, list):
            raise RuntimeError("Local embedding model returned invalid data")
        vector = embedding[0] if isinstance(embedding[0], list) else embedding
        return list(vector)

    def _normalize_embedding_length(
        self,
        embedding: List[float],
        source: str,
        allow_resize_in_production: bool = False,
    ) -> List[float]:
        if len(embedding) == self.embedding_dim:
            return embedding
        message = (
            f"Embedding length {len(embedding)} from {source} does not match "
            f"expected {self.embedding_dim}."
        )
        if not allow_resize_in_production and (self.require_embeddings or self._is_production()):
            raise RuntimeError(message)
        self.log(f"{message} Resizing to expected size.", "WARNING")
        return self._resize_embedding(embedding)

    def _resize_embedding(self, embedding: List[float]) -> List[float]:
        if len(embedding) == self.embedding_dim:
            return list(embedding)
        if not embedding:
            return [0.0] * self.embedding_dim
        if len(embedding) < self.embedding_dim:
            padded = np.pad(
                np.asarray(embedding, dtype=float),
                (0, self.embedding_dim - len(embedding)),
                mode="constant",
            )
            return padded.tolist()
        original = np.asarray(embedding, dtype=float)
        original_indices = np.linspace(0.0, 1.0, num=len(original))
        target_indices = np.linspace(0.0, 1.0, num=self.embedding_dim)
        resized = np.interp(target_indices, original_indices, original)
        return resized.tolist()

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

        while len(embedding) < self.embedding_dim:
            embedding.extend(embedding[:min(len(embedding), self.embedding_dim - len(embedding))])

        return embedding[: self.embedding_dim]


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
