#!/usr/bin/env python3
"""
Variation Asset Pipeline - End-to-End SimReady Asset Generation

This job orchestrates the complete pipeline for generating variation assets
needed for domain randomization in Isaac Sim Replicator:

1. Read manifest from replicator-job (variation_assets/manifest.json)
2. Check asset library for existing assets (optional optimization)
3. For missing assets:
   a. Generate reference images using Gemini 3.0 Pro Image (Nano Banana Pro)
   b. Convert to 3D using SAM3D (fast), Hunyuan (quality), or UltraShape (ultra)
   c. Add physics properties inline (mass, friction, COM, collision)
4. Output: SimReady USDZ assets ready for Replicator placement

3D Backend Options:
    - SAM3D: Fast, good for simple objects (30-60 seconds per asset)
    - Hunyuan: High quality, slower (2-5 minutes per asset)
    - UltraShape: Highest quality via Hunyuan + diffusion refinement (5-10 minutes per asset)
      Reference: https://github.com/PKU-YuanGroup/UltraShape-1.0

Pipeline Flow:
    replicator-job manifest.json
            ↓
    variation-asset-pipeline-job
            ↓
    variation_assets/*.usdz (SimReady)

Configuration (Environment Variables):
    SCENE_ID: Scene identifier (required)
    BUCKET: GCS bucket name
    REPLICATOR_PREFIX: Path to replicator bundle (default: scenes/{SCENE_ID}/replicator)
    VARIATION_ASSETS_PREFIX: Output path (default: scenes/{SCENE_ID}/variation_assets)

    3D_BACKEND: "sam3d" | "hunyuan" | "ultrashape" | "auto" (default: "auto")
    QUALITY_MODE: "fast" | "balanced" | "quality" | "ultra" (default: "balanced")
    MAX_ASSETS: Maximum number of assets to generate (default: all)
    PRIORITY_FILTER: "required" | "recommended" | "optional" | "" (default: "")

    ASSET_LIBRARY_PATH: Path to shared asset library (optional)
    SKIP_EXISTING: Skip assets that already exist (default: "1")
    DRY_RUN: Skip actual generation (default: "0")

    # UltraShape-specific configuration
    ULTRASHAPE_REPO_PATH: Path to UltraShape repository (default: /app/UltraShape-1.0)
    ULTRASHAPE_CHECKPOINT: Path to UltraShape checkpoint (default: /app/ultrashape/checkpoints/ultrashape_v1.pt)
    ULTRASHAPE_CONFIG: Path to UltraShape config (default: /app/ultrashape/configs/infer_dit_refine.yaml)
"""

import json
import os
import sys
import time
import datetime
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

from tools.validation.entrypoint_checks import validate_required_env_vars
import io
import base64
import filecmp

from PIL import Image
import numpy as np

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[VAR-PIPELINE] ERROR: google-genai package not installed", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Constants
# ============================================================================

GCS_ROOT = Path("/mnt/gcs")

# usd_from_gltf timeout (seconds)
USD_FROM_GLTF_TIMEOUT_S = float(os.getenv("USD_FROM_GLTF_TIMEOUT_S", "300"))

# Gemini model for image generation (Nano Banana Pro / Gemini 3.0 Pro Image Preview)
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"

# Quality presets
QUALITY_PRESETS = {
    "fast": {
        "3d_backend": "sam3d",
        "image_size": 512,
        "sam3d_normalize": True,
        "hunyuan_render_size": 512,
        "hunyuan_texture_size": 1024,
        "ultrashape_steps": 30,
        "ultrashape_octree_resolution": 128,
    },
    "balanced": {
        "3d_backend": "auto",  # SAM3D for simple, Hunyuan for complex
        "image_size": 1024,
        "sam3d_normalize": True,
        "hunyuan_render_size": 1024,
        "hunyuan_texture_size": 2048,
        "ultrashape_steps": 50,
        "ultrashape_octree_resolution": 256,
    },
    "quality": {
        "3d_backend": "hunyuan",
        "image_size": 2048,
        "sam3d_normalize": True,
        "hunyuan_render_size": 2048,
        "hunyuan_texture_size": 4096,
        "ultrashape_steps": 50,
        "ultrashape_octree_resolution": 256,
    },
    "ultra": {
        "3d_backend": "ultrashape",  # Hunyuan + UltraShape refinement
        "image_size": 2048,
        "sam3d_normalize": True,
        "hunyuan_render_size": 2048,
        "hunyuan_texture_size": 4096,
        "ultrashape_steps": 75,
        "ultrashape_octree_resolution": 384,
    },
}

# Category-based complexity hints for auto backend selection
SIMPLE_CATEGORIES = {"dishes", "utensils", "cans", "bottles", "boxes", "containers"}
COMPLEX_CATEGORIES = {"clothing", "food", "produce", "tools", "electronics", "lab_equipment"}
GENERIC_CATEGORIES = {"", "generic", "misc", "miscellaneous", "unknown", "other"}

COMPLEXITY_KEYWORDS = {
    "intricate": 2,
    "detailed": 2,
    "complex": 2,
    "layered": 2,
    "mechanical": 2,
    "electronics": 2,
    "circuit": 2,
    "buttons": 1,
    "knobs": 1,
    "switches": 1,
    "hinges": 1,
    "folds": 1,
    "fabric": 1,
    "textured": 1,
    "patterned": 1,
    "transparent": 1,
    "reflective": 1,
    "glass": 1,
    "metallic": 1,
    "multiple parts": 2,
}
COMPLEXITY_SCORE_THRESHOLD = 2

# Physics defaults by category
PHYSICS_DEFAULTS = {
    "dishes": {
        "bulk_density": 2400,  # ceramic
        "static_friction": 0.4,
        "dynamic_friction": 0.35,
        "restitution": 0.1,
        "collision_shape": "convex",
    },
    "utensils": {
        "bulk_density": 7800,  # stainless steel
        "static_friction": 0.3,
        "dynamic_friction": 0.25,
        "restitution": 0.15,
        "collision_shape": "convex",
    },
    "groceries": {
        "bulk_density": 800,  # packaged goods (lots of air)
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.05,
        "collision_shape": "box",
    },
    "clothing": {
        "bulk_density": 300,  # fabric (very airy)
        "static_friction": 0.7,
        "dynamic_friction": 0.6,
        "restitution": 0.02,
        "collision_shape": "convex",
    },
    "bottles": {
        "bulk_density": 1200,  # glass/plastic with liquid
        "static_friction": 0.35,
        "dynamic_friction": 0.3,
        "restitution": 0.1,
        "collision_shape": "convex",
    },
    "cans": {
        "bulk_density": 1500,  # metal with contents
        "static_friction": 0.4,
        "dynamic_friction": 0.35,
        "restitution": 0.1,
        "collision_shape": "convex",
    },
    "boxes": {
        "bulk_density": 400,  # cardboard with contents
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.05,
        "collision_shape": "box",
    },
    "tools": {
        "bulk_density": 3500,  # mixed metal/plastic
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.1,
        "collision_shape": "convex",
    },
    "default": {
        "bulk_density": 600,
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.1,
        "collision_shape": "convex",
    },
}


# ============================================================================
# Data Classes
# ============================================================================

class Backend3D(str, Enum):
    SAM3D = "sam3d"
    HUNYUAN = "hunyuan"
    ULTRASHAPE = "ultrashape"  # Hunyuan + UltraShape refinement
    AUTO = "auto"


@dataclass
class AssetSpec:
    """Specification for a variation asset."""
    name: str
    category: str
    description: str
    semantic_class: str
    priority: str
    source_hint: Optional[str] = None
    example_variants: List[str] = field(default_factory=list)
    physics_hints: Dict[str, Any] = field(default_factory=dict)
    material_hint: Optional[str] = None
    style_hint: Optional[str] = None
    generation_prompt_hint: Optional[str] = None


@dataclass
class AssetResult:
    """Result of processing a single asset."""
    name: str
    success: bool
    stage_completed: str  # "skipped", "library", "image", "3d", "physics", "complete"
    output_path: Optional[str] = None
    asset_dir: Optional[str] = None
    reference_image: Optional[str] = None
    metadata_path: Optional[str] = None
    error: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    scene_id: str
    bucket: Optional[str]
    replicator_prefix: str
    variation_assets_prefix: str
    backend_3d: Backend3D
    quality_mode: str
    max_assets: Optional[int]
    priority_filter: Optional[str]
    asset_library_path: Optional[str]
    registry_backend_uri: Optional[str]
    register_assets: bool
    skip_existing: bool
    dry_run: bool
    vector_backend_uri: Optional[str]
    vector_similarity_threshold: float
    vector_max_candidates: int


@dataclass
class EmbeddingRecord:
    """Embedding record for an asset in the vector store."""

    asset_id: str
    category: str
    path: str
    embedding: List[float]
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None


class VectorIndex:
    """Lightweight in-memory vector index using cosine similarity."""

    def __init__(self, records: List[EmbeddingRecord]):
        self.records = records

    def search(
        self,
        query_embedding: List[float],
        category: str,
        top_k: int,
    ) -> List[Tuple[float, EmbeddingRecord]]:
        if not self.records:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        scored: List[Tuple[float, EmbeddingRecord]] = []

        for record in self.records:
            if record.category != category:
                continue
            v = np.array(record.embedding, dtype=np.float32)
            v_norm = v / (np.linalg.norm(v) + 1e-8)
            similarity = float(np.dot(q_norm, v_norm))
            scored.append((similarity, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


def resolve_vector_backend_uri(
    explicit_uri: Optional[str], library_path: Optional[str]
) -> Optional[str]:
    """Resolve vector backend URI, defaulting to index.json in library if present."""

    if explicit_uri:
        return explicit_uri

    if library_path:
        default_index = GCS_ROOT / library_path / "index.json"
        if default_index.is_file():
            return f"file://{default_index}"

    return None


def load_vector_index(
    backend_uri: Optional[str], library_path: Optional[str]
) -> Optional[VectorIndex]:
    """
    Load vector index from backend URI.

    Currently supports file:// URIs pointing to JSON with structure:
    {
        "assets": [
            {
                "asset_id": "plate_white_001",
                "category": "dishes",
                "path": "dishes/plate_white_001.usdz",
                "embedding": [...],
                "tags": ["ceramic", "plate"],
                "description": "white ceramic plate"
            }
        ]
    }
    """

    if not backend_uri:
        return None

    if backend_uri.startswith("file://"):
        index_path = Path(backend_uri[len("file://") :])
    else:
        index_path = Path(backend_uri)

    if not index_path.is_file():
        print(
            f"[VAR-PIPELINE] Vector index not found at {index_path}; skipping vector search"
        )
        return None

    try:
        with index_path.open("r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[VAR-PIPELINE] Failed to load vector index: {e}")
        return None

    records: List[EmbeddingRecord] = []
    for entry in data.get("assets", []):
        embedding = entry.get("embedding")
        if not embedding:
            continue
        records.append(
            EmbeddingRecord(
                asset_id=entry.get("asset_id", entry.get("id", "")),
                category=entry.get("category", ""),
                path=entry.get("path", ""),
                embedding=embedding,
                tags=entry.get("tags", []),
                description=entry.get("description"),
            )
        )

    print(
        f"[VAR-PIPELINE] Loaded vector index with {len(records)} records from {index_path}"
    )

    return VectorIndex(records)


# ============================================================================
# Asset Registry & Library Ingestion
# ============================================================================


def build_storage_uri(path: Path, bucket: Optional[str]) -> str:
    """Convert a local GCS-mounted path to a gs:// URI when possible."""

    try:
        rel = path.relative_to(GCS_ROOT)
        if bucket:
            return f"gs://{bucket}/{rel.as_posix()}"
        return rel.as_posix()
    except ValueError:
        return str(path)


def is_same_file(src: Path, dst: Path) -> bool:
    try:
        return dst.is_file() and filecmp.cmp(src, dst, shallow=False)
    except OSError:
        return False


class AssetRegistry:
    """Simple registry backend with idempotent file-based upsert."""

    def __init__(self, backend_uri: Optional[str], enabled: bool, dry_run: bool):
        self.backend_uri = backend_uri
        self.enabled = enabled and backend_uri is not None
        self.dry_run = dry_run

    def _resolve_path(self) -> Optional[Path]:
        if not self.backend_uri:
            return None

        if self.backend_uri.startswith("file://"):
            return Path(self.backend_uri[len("file://") :])

        # Default to local path if no scheme
        return Path(self.backend_uri)

    def upsert(self, entry: Dict[str, Any]) -> None:
        if not self.enabled:
            print("[VAR-PIPELINE] Registry disabled; skipping registry update")
            return

        if self.dry_run:
            print("[VAR-PIPELINE] [DRY-RUN] Would update registry")
            return

        registry_path = self._resolve_path()
        if registry_path is None:
            print("[VAR-PIPELINE] Registry backend not configured; skipping")
            return

        registry_path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {"assets": []}
        try:
            if registry_path.is_file():
                with registry_path.open("r") as f:
                    data = json.load(f)
        except Exception as e:
            print(f"[VAR-PIPELINE] Failed to read registry {registry_path}: {e}")
            data = {"assets": []}

        assets: List[Dict[str, Any]] = data.get("assets", [])
        existing_idx = next(
            (i for i, a in enumerate(assets) if a.get("asset_id") == entry["asset_id"]),
            None,
        )

        if existing_idx is not None and assets[existing_idx] == entry:
            print(
                f"[VAR-PIPELINE] Registry already up-to-date for {entry['asset_id']}"
            )
            return

        if existing_idx is not None:
            assets[existing_idx] = entry
            print(f"[VAR-PIPELINE] Updated registry entry for {entry['asset_id']}")
        else:
            assets.append(entry)
            print(f"[VAR-PIPELINE] Added registry entry for {entry['asset_id']}")

        data["assets"] = assets

        try:
            with registry_path.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[VAR-PIPELINE] Failed to write registry {registry_path}: {e}")


def persist_asset_to_library(
    asset: AssetSpec, result: AssetResult, config: PipelineConfig
) -> Dict[str, Optional[Path]]:
    """
    Copy generated assets into the shared library structure.

    Returns paths for the published asset, metadata, and thumbnail (if written).
    """

    if not config.asset_library_path:
        return {}

    if config.dry_run:
        print("[VAR-PIPELINE] [DRY-RUN] Would publish asset to library")
        return {}

    if not result.output_path:
        return {}

    lib_root = GCS_ROOT / config.asset_library_path
    lib_category_dir = lib_root / asset.category
    lib_category_dir.mkdir(parents=True, exist_ok=True)

    source_asset_path = Path(result.output_path)
    asset_safe = safe_name(asset.name)

    # Skip copying if asset already inside the library root
    try:
        source_asset_path.relative_to(lib_root)
        asset_in_library = True
    except ValueError:
        asset_in_library = False

    dest_asset_path = lib_category_dir / f"{asset_safe}{source_asset_path.suffix or '.usdz'}"

    if not asset_in_library:
        if dest_asset_path.suffix != source_asset_path.suffix:
            dest_asset_path = lib_category_dir / f"{asset_safe}{source_asset_path.suffix}"

        if dest_asset_path.exists() and is_same_file(source_asset_path, dest_asset_path):
            print(
                f"[VAR-PIPELINE] Library already has up-to-date asset for {asset.name}"
            )
        else:
            shutil.copy(source_asset_path, dest_asset_path)
            print(
                f"[VAR-PIPELINE] Published asset to library: {dest_asset_path}"
            )

    # Metadata
    metadata_candidates = []
    if result.metadata_path:
        metadata_candidates.append(Path(result.metadata_path))
    if result.asset_dir:
        metadata_candidates.append(Path(result.asset_dir) / "metadata.json")

    dest_metadata_path = None
    for meta in metadata_candidates:
        if meta.is_file():
            dest_metadata_path = lib_category_dir / f"{asset_safe}.json"
            if not (dest_metadata_path.exists() and is_same_file(meta, dest_metadata_path)):
                shutil.copy(meta, dest_metadata_path)
                print(
                    f"[VAR-PIPELINE] Published metadata for {asset.name} -> {dest_metadata_path}"
                )
            break

    # Thumbnail
    thumbnail_candidates = []
    if result.reference_image:
        thumbnail_candidates.append(Path(result.reference_image))
    if result.asset_dir:
        thumbnail_candidates.append(Path(result.asset_dir) / "reference.png")

    dest_thumbnail_path = None
    for thumb in thumbnail_candidates:
        if thumb.is_file():
            dest_thumbnail_path = lib_category_dir / f"{asset_safe}_thumb{thumb.suffix}"
            if not (dest_thumbnail_path.exists() and is_same_file(thumb, dest_thumbnail_path)):
                shutil.copy(thumb, dest_thumbnail_path)
                print(
                    f"[VAR-PIPELINE] Published thumbnail for {asset.name} -> {dest_thumbnail_path}"
                )
            break

    return {
        "asset": dest_asset_path if dest_asset_path.exists() else None,
        "metadata": dest_metadata_path,
        "thumbnail": dest_thumbnail_path,
    }


def ingest_asset(
    asset: AssetSpec,
    result: AssetResult,
    config: PipelineConfig,
    registry: AssetRegistry,
    source_pipeline: str = "variation",
):
    """Persist assets to the library and update registry with idempotency."""

    if config.dry_run:
        print("[VAR-PIPELINE] [DRY-RUN] Skipping ingestion")
        return

    published_paths = persist_asset_to_library(asset, result, config)

    if not config.register_assets:
        return

    storage_uris: Dict[str, str] = {}
    if result.output_path:
        storage_uris["variation_output"] = build_storage_uri(
            Path(result.output_path), config.bucket
        )

    for key, path in published_paths.items():
        if path:
            storage_uris[f"library_{key}"] = build_storage_uri(path, config.bucket)

    metadata_uri = None
    if published_paths.get("metadata"):
        metadata_uri = build_storage_uri(published_paths["metadata"], config.bucket)
    elif result.metadata_path:
        metadata_uri = build_storage_uri(Path(result.metadata_path), config.bucket)

    thumbnail_uri = None
    if published_paths.get("thumbnail"):
        thumbnail_uri = build_storage_uri(published_paths["thumbnail"], config.bucket)
    elif result.reference_image:
        thumbnail_uri = build_storage_uri(Path(result.reference_image), config.bucket)

    entry = {
        "asset_id": safe_name(asset.name),
        "category": asset.category,
        "tags": list({asset.category, asset.semantic_class, *asset.example_variants}),
        "source_pipeline": source_pipeline,
        "storage_uris": storage_uris,
        "metadata_uri": metadata_uri,
        "thumbnail_uri": thumbnail_uri,
        "simready": True,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "priority": asset.priority,
    }

    registry.upsert(entry)


# ============================================================================
# Utility Functions
# ============================================================================

def getenv_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def safe_name(name: str) -> str:
    """Convert name to filesystem-safe string."""
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def create_gemini_client():
    """Create Gemini client using API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def build_asset_embedding_text(asset: AssetSpec) -> str:
    """Create a descriptive string for embedding generation."""

    sections = [
        f"name: {asset.name}",
        f"category: {asset.category}",
        f"description: {asset.description}",
        f"semantic_class: {asset.semantic_class}",
        f"priority: {asset.priority}",
    ]

    if asset.example_variants:
        sections.append(f"variants: {', '.join(asset.example_variants)}")
    if asset.material_hint:
        sections.append(f"materials: {asset.material_hint}")
    if asset.style_hint:
        sections.append(f"style: {asset.style_hint}")

    return " | ".join(sections)


def embed_asset_spec(asset: AssetSpec, client) -> Optional[List[float]]:
    """Embed asset description using Gemini embeddings."""

    if client is None:
        return None

    try:
        response = client.models.embed_content(
            model="text-embedding-004", contents=build_asset_embedding_text(asset)
        )

        embedding = None
        if hasattr(response, "embedding"):
            embedding = getattr(response.embedding, "values", response.embedding)
        elif hasattr(response, "values"):
            embedding = response.values

        if embedding:
            return list(embedding)
    except Exception as e:
        print(f"[VAR-PIPELINE] Failed to embed asset spec: {e}")

    return None


# ============================================================================
# Asset Library (Optional)
# ============================================================================

def check_asset_library(
    asset: AssetSpec,
    library_path: Optional[str],
    vector_index: Optional[VectorIndex],
    config: PipelineConfig,
    client,
) -> Optional[Path]:
    """
    Check if an asset exists in the shared library.

    Library structure:
        {library_path}/{category}/{asset_name}.usdz
        {library_path}/{category}/{asset_name}_meta.json
    """
    if not library_path:
        return None

    lib_root = GCS_ROOT / library_path
    if not lib_root.is_dir():
        return None

    # Check by exact name
    asset_path = lib_root / asset.category / f"{safe_name(asset.name)}.usdz"
    if asset_path.is_file():
        print(f"[VAR-PIPELINE] Found in library: {asset.name} -> {asset_path}")
        return asset_path

    # Semantic/vector search fallback
    backend_uri = resolve_vector_backend_uri(
        config.vector_backend_uri, library_path
    )

    if not backend_uri or not vector_index:
        return None

    if config.vector_max_candidates <= 0:
        return None

    query_embedding = embed_asset_spec(asset, client)
    if not query_embedding:
        return None

    candidates = vector_index.search(
        query_embedding=query_embedding,
        category=asset.category,
        top_k=config.vector_max_candidates,
    )

    if not candidates:
        print(
            f"[VAR-PIPELINE] No vector candidates for {asset.name} in category {asset.category}"
        )
        return None

    best_score, best_record = candidates[0]
    best_path = Path(best_record.path)
    if not best_path.is_absolute():
        best_path = lib_root / best_path

    print(
        f"[VAR-PIPELINE] Vector match for {asset.name}: {best_record.asset_id} -> "
        f"{best_path} (score={best_score:.3f})"
    )

    if best_score >= config.vector_similarity_threshold and best_path.is_file():
        return best_path

    return None


def copy_from_library(
    library_asset_path: Path,
    output_dir: Path,
    asset_name: str
) -> Path:
    """Copy asset from library to output directory."""
    output_path = output_dir / f"{safe_name(asset_name)}.usdz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(library_asset_path, output_path)

    # Also copy metadata if exists
    meta_path = library_asset_path.with_suffix('.json')
    if meta_path.is_file():
        shutil.copy(meta_path, output_path.with_suffix('.json'))

    thumb_path = library_asset_path.with_name(f"{library_asset_path.stem}_thumb.png")
    if thumb_path.is_file():
        shutil.copy(thumb_path, output_path.with_name(f"{output_path.stem}_thumb.png"))

    return output_path


# ============================================================================
# Image Generation (Gemini 3.0 Pro Image / Nano Banana Pro)
# ============================================================================

def build_image_generation_prompt(asset: AssetSpec, scene_context: str = "") -> str:
    """
    Build prompt for Gemini to generate a product photography style image.
    """
    # Use generation_prompt_hint if available
    if asset.generation_prompt_hint:
        base_description = asset.generation_prompt_hint
    else:
        base_description = f"a {asset.description}"

    # Material and style hints
    material = asset.material_hint or "appropriate realistic materials"
    style = asset.style_hint or "photorealistic product photography"

    prompt = f"""Generate a professional product photography image of {base_description}.

Style Requirements:
- {style}
- Materials: {material}
- Single isolated object, centered in frame
- Pure white or light gray studio background
- Soft, even studio lighting (3-point lighting setup)
- Front-facing view with slight elevation (15-20 degrees from front)
- Object fills approximately 70-80% of frame
- Sharp focus throughout, high detail
- No shadows on background (floating appearance)
- Photorealistic quality suitable for 3D reconstruction
- Square aspect ratio (1:1)

The image should look like a professional product catalog photo suitable for converting to a 3D model."""

    return prompt


def generate_reference_image_gemini(
    client,
    asset: AssetSpec,
    output_dir: Path,
    scene_context: str = "",
    dry_run: bool = False
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Generate a reference image using Gemini 3.0 Pro Image native image generation.

    Returns: (success, image_path, error_message)
    """
    asset_dir = output_dir / safe_name(asset.name)
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_path = asset_dir / "reference.png"

    if dry_run:
        print(f"[VAR-PIPELINE] [DRY-RUN] Would generate image for: {asset.name}")
        return True, image_path, None

    prompt = build_image_generation_prompt(asset, scene_context)

    print(f"[VAR-PIPELINE] Generating image for: {asset.name}")

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
        # Use Gemini 3.0 Pro Image with native image generation
            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    temperature=0.8,
                ),
            )

            # Extract image from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Decode base64 image data
                            image_data = part.inline_data.data
                            if isinstance(image_data, str):
                                image_bytes = base64.b64decode(image_data)
                            else:
                                image_bytes = image_data

                            # Save image
                            img = Image.open(io.BytesIO(image_bytes))
                            img.save(str(image_path), format='PNG')
                            print(f"[VAR-PIPELINE] Generated image: {asset.name} -> {image_path}")
                            return True, image_path, None

            raise ValueError("No image data in response")

        except Exception as e:
            last_error = str(e)
            print(f"[VAR-PIPELINE] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff

    return False, None, last_error


# ============================================================================
# 3D Conversion (SAM3D / Hunyuan)
# ============================================================================

def score_complexity_from_keywords(text: str) -> Tuple[int, List[str]]:
    """Score complexity based on keyword hits in the provided text."""
    if not text:
        return 0, []
    text_lower = text.lower()
    score = 0
    hits = []
    for keyword, weight in COMPLEXITY_KEYWORDS.items():
        if keyword in text_lower:
            score += weight
            hits.append(keyword)
    return score, hits


def build_generic_complexity_text(asset: AssetSpec) -> str:
    """Build a text bundle for complexity scoring when category is generic."""
    parts = [
        asset.description,
        asset.semantic_class,
        asset.material_hint,
        asset.generation_prompt_hint,
    ]
    return " ".join(part for part in parts if part)


def select_3d_backend(asset: AssetSpec, config: PipelineConfig) -> Tuple[str, Dict[str, Any]]:
    """
    Select 3D backend based on asset complexity and config.
    """
    category_lower = asset.category.strip().lower()
    decision_metadata = {
        "category": asset.category,
        "keyword_hits": [],
        "complexity_score": 0,
    }

    if config.backend_3d != Backend3D.AUTO:
        decision_metadata["selection"] = config.backend_3d.value
        return config.backend_3d.value, decision_metadata

    # Auto selection based on category
    if category_lower in SIMPLE_CATEGORIES:
        decision_metadata["selection"] = "sam3d"
        return "sam3d", decision_metadata
    if category_lower in COMPLEX_CATEGORIES:
        decision_metadata["selection"] = "hunyuan"
        return "hunyuan", decision_metadata

    # For generic/empty categories, use keyword-based scoring on descriptive fields
    if category_lower in GENERIC_CATEGORIES:
        complexity_text = build_generic_complexity_text(asset)
        score, hits = score_complexity_from_keywords(complexity_text)
        decision_metadata["complexity_score"] = score
        decision_metadata["keyword_hits"] = hits
        backend = "hunyuan" if score >= COMPLEXITY_SCORE_THRESHOLD else "sam3d"
        decision_metadata["selection"] = backend
        return backend, decision_metadata

    # Default to SAM3D for speed
    decision_metadata["selection"] = "sam3d"
    return "sam3d", decision_metadata


def run_sam3d_reconstruction(
    image_path: Path,
    output_dir: Path,
    asset_name: str,
    normalize: bool = True,
    dry_run: bool = False
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Run SAM3D reconstruction on a reference image.

    This is a simplified inline version that uses the SAM3D inference directly.
    In production, this would use the 3D-RE-GEN pipeline or Inference class.

    Returns: (success, glb_path, error_message)
    """
    if dry_run:
        print(f"[VAR-PIPELINE] [DRY-RUN] Would run SAM3D for: {asset_name}")
        return True, output_dir / safe_name(asset_name) / "model.glb", None

    try:
        # Try to import SAM3D inference
        try:
            from inference import Inference
            SAM3D_AVAILABLE = True
        except ImportError:
            SAM3D_AVAILABLE = False

        if not SAM3D_AVAILABLE:
            return False, None, "SAM3D inference not available in this environment"

        # Find SAM3D config
        config_candidates = [
            Path("/app/sam3d-objects/checkpoints/hf/pipeline.yaml"),
            Path("/mnt/gcs/sam3d/checkpoints/hf/pipeline.yaml"),
            Path("/workspace/sam3d-objects/checkpoints/hf/pipeline.yaml"),
        ]

        config_path = None
        for candidate in config_candidates:
            if candidate.is_file():
                config_path = candidate
                break

        if config_path is None:
            return False, None, "SAM3D config not found"

        print(f"[VAR-PIPELINE] Running SAM3D on: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        rgb = np.array(image)

        # Simple background mask (white background)
        gray = rgb.mean(axis=-1)
        mask = (np.abs(gray - 240) > 10).astype(np.uint8)

        # Run inference
        inference = Inference(str(config_path), compile=False)
        output = inference(rgb, mask, seed=42)

        # Save mesh
        mesh = output.get("mesh")
        if mesh is None:
            return False, None, "SAM3D returned no mesh"

        asset_out_dir = output_dir / safe_name(asset_name)
        asset_out_dir.mkdir(parents=True, exist_ok=True)

        glb_path = asset_out_dir / "model.glb"
        mesh.export(str(glb_path))

        print(f"[VAR-PIPELINE] SAM3D complete: {asset_name} -> {glb_path}")
        return True, glb_path, None

    except Exception as e:
        return False, None, f"SAM3D error: {str(e)}"


def run_hunyuan_reconstruction(
    image_path: Path,
    output_dir: Path,
    asset_name: str,
    render_size: int = 1024,
    texture_size: int = 2048,
    dry_run: bool = False
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Run Hunyuan3D reconstruction on a reference image.

    Returns: (success, glb_path, error_message)
    """
    if dry_run:
        print(f"[VAR-PIPELINE] [DRY-RUN] Would run Hunyuan for: {asset_name}")
        return True, output_dir / safe_name(asset_name) / "model.glb", None

    try:
        # Try to import Hunyuan components
        HUNYUAN_REPO_ROOT = Path(os.getenv("HUNYUAN_REPO_ROOT", "/app/Hunyuan3D-2.1"))
        sys.path.insert(0, str(HUNYUAN_REPO_ROOT))
        sys.path.insert(0, str(HUNYUAN_REPO_ROOT / "hy3dshape"))
        sys.path.insert(0, str(HUNYUAN_REPO_ROOT / "hy3dpaint"))

        try:
            from hy3dshape.rembg import BackgroundRemover
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            HUNYUAN_AVAILABLE = True
        except ImportError:
            HUNYUAN_AVAILABLE = False

        if not HUNYUAN_AVAILABLE:
            return False, None, "Hunyuan3D not available in this environment"

        print(f"[VAR-PIPELINE] Running Hunyuan3D on: {image_path}")

        # Load model
        model_path = os.getenv("HUNYUAN_MODEL_PATH", "tencent/Hunyuan3D-2.1")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

        # Load and prepare image
        image = Image.open(image_path)
        rembg = BackgroundRemover()

        if image.mode == "RGB":
            image = rembg(image)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Generate mesh
        meshes = pipeline(image=image, num_inference_steps=50)

        # Handle output format
        if isinstance(meshes, list):
            first = meshes[0] if meshes else None
            if isinstance(first, list):
                mesh = first[0] if first else None
            else:
                mesh = first
        else:
            mesh = meshes

        if mesh is None:
            return False, None, "Hunyuan returned no mesh"

        # Save mesh
        asset_out_dir = output_dir / safe_name(asset_name)
        asset_out_dir.mkdir(parents=True, exist_ok=True)

        glb_path = asset_out_dir / "model.glb"
        mesh.export(str(glb_path))

        print(f"[VAR-PIPELINE] Hunyuan3D complete: {asset_name} -> {glb_path}")
        return True, glb_path, None

    except Exception as e:
        return False, None, f"Hunyuan error: {str(e)}"


def run_ultrashape_reconstruction(
    image_path: Path,
    output_dir: Path,
    asset_name: str,
    render_size: int = 2048,
    texture_size: int = 4096,
    ultrashape_steps: int = 50,
    ultrashape_octree_resolution: int = 256,
    dry_run: bool = False
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Run UltraShape reconstruction on a reference image.

    UltraShape is a two-stage pipeline:
    1. First generates a coarse mesh using Hunyuan3D-2.1
    2. Then refines the mesh using UltraShape diffusion model

    This produces the highest quality 3D reconstructions, but is slower.

    Reference: https://github.com/PKU-YuanGroup/UltraShape-1.0

    Returns: (success, glb_path, error_message)
    """
    if dry_run:
        print(f"[VAR-PIPELINE] [DRY-RUN] Would run UltraShape for: {asset_name}")
        return True, output_dir / safe_name(asset_name) / "model.glb", None

    try:
        # Stage 1: Generate coarse mesh with Hunyuan3D
        print(f"[VAR-PIPELINE] UltraShape Stage 1: Generating coarse mesh with Hunyuan3D")

        success, coarse_mesh_path, error = run_hunyuan_reconstruction(
            image_path=image_path,
            output_dir=output_dir,
            asset_name=f"{asset_name}_coarse",
            render_size=render_size,
            texture_size=texture_size,
            dry_run=False,
        )

        if not success:
            return False, None, f"UltraShape Stage 1 (Hunyuan) failed: {error}"

        # Stage 2: Refine with UltraShape
        print(f"[VAR-PIPELINE] UltraShape Stage 2: Refining mesh with UltraShape")

        try:
            # Try to import UltraShape refiner
            ULTRASHAPE_REPO_PATH = Path(os.getenv("ULTRASHAPE_REPO_PATH", "/app/UltraShape-1.0"))
            sys.path.insert(0, str(ULTRASHAPE_REPO_PATH))

            # Add the project's ultrashape module
            project_ultrashape = Path(__file__).parent.parent / "ultrashape"
            if project_ultrashape.is_dir():
                sys.path.insert(0, str(project_ultrashape.parent))

            from ultrashape.ultrashape_refiner import UltraShapeRefiner, UltraShapeConfig

            # Configure UltraShape
            config = UltraShapeConfig(
                checkpoint_path=os.getenv(
                    "ULTRASHAPE_CHECKPOINT",
                    "/app/ultrashape/checkpoints/ultrashape_v1.pt"
                ),
                config_path=os.getenv(
                    "ULTRASHAPE_CONFIG",
                    "/app/ultrashape/configs/infer_dit_refine.yaml"
                ),
                diffusion_steps=ultrashape_steps,
                octree_resolution=ultrashape_octree_resolution,
            )

            refiner = UltraShapeRefiner(
                config=config,
                ultrashape_repo_path=str(ULTRASHAPE_REPO_PATH),
            )

            # Check if UltraShape is available
            if not refiner.is_available():
                print("[VAR-PIPELINE] UltraShape not available, using Hunyuan output directly")
                # Rename coarse mesh to final output
                asset_out_dir = output_dir / safe_name(asset_name)
                asset_out_dir.mkdir(parents=True, exist_ok=True)
                final_glb_path = asset_out_dir / "model.glb"

                import shutil
                shutil.copy(coarse_mesh_path, final_glb_path)

                return True, final_glb_path, None

            # Run refinement
            asset_out_dir = output_dir / safe_name(asset_name)
            asset_out_dir.mkdir(parents=True, exist_ok=True)
            refined_glb_path = asset_out_dir / "model.glb"

            success, output_path, error = refiner.refine(
                image_path=image_path,
                coarse_mesh_path=coarse_mesh_path,
                output_path=refined_glb_path,
                diffusion_steps=ultrashape_steps,
            )

            if not success:
                print(f"[VAR-PIPELINE] UltraShape refinement failed: {error}")
                print("[VAR-PIPELINE] Falling back to Hunyuan output")

                import shutil
                shutil.copy(coarse_mesh_path, refined_glb_path)
                return True, refined_glb_path, None

            # Cleanup coarse mesh
            try:
                coarse_dir = output_dir / safe_name(f"{asset_name}_coarse")
                if coarse_dir.is_dir():
                    import shutil
                    shutil.rmtree(coarse_dir)
            except Exception:
                print(
                    f"[VAR-PIPELINE] WARNING: Failed to clean up coarse directory {coarse_dir}",
                    file=sys.stderr,
                )

            print(f"[VAR-PIPELINE] UltraShape complete: {asset_name} -> {refined_glb_path}")
            return True, refined_glb_path, None

        except ImportError as e:
            print(f"[VAR-PIPELINE] UltraShape not available ({e}), using Hunyuan output")
            # Copy coarse mesh to final output
            asset_out_dir = output_dir / safe_name(asset_name)
            asset_out_dir.mkdir(parents=True, exist_ok=True)
            final_glb_path = asset_out_dir / "model.glb"

            import shutil
            shutil.copy(coarse_mesh_path, final_glb_path)

            return True, final_glb_path, None

    except Exception as e:
        return False, None, f"UltraShape error: {str(e)}"


def convert_glb_to_usdz(glb_path: Path, usdz_path: Path) -> bool:
    """Convert GLB to USDZ using usd_from_gltf if available."""
    usd_from_gltf = shutil.which("usd_from_gltf")
    if usd_from_gltf is None:
        print("[VAR-PIPELINE] WARNING: usd_from_gltf not available", file=sys.stderr)
        return False

    usdz_path.parent.mkdir(parents=True, exist_ok=True)
    command = [usd_from_gltf, str(glb_path), "-o", str(usdz_path)]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=USD_FROM_GLTF_TIMEOUT_S,
        )
        return True
    except subprocess.TimeoutExpired:
        command_str = " ".join(command)
        print(
            "[VAR-PIPELINE] ERROR: usd_from_gltf timed out after "
            f"{USD_FROM_GLTF_TIMEOUT_S}s: {command_str}",
            file=sys.stderr,
        )
        return False
    except subprocess.CalledProcessError as e:
        details = []
        if e.stderr:
            details.append(f"stderr: {e.stderr.strip()}")
        if e.stdout:
            details.append(f"stdout: {e.stdout.strip()}")
        detail_str = f" ({'; '.join(details)})" if details else ""
        print(
            f"[VAR-PIPELINE] usd_from_gltf failed with exit code {e.returncode}"
            f"{detail_str}",
            file=sys.stderr,
        )
        return False


# ============================================================================
# Physics Properties (Inline SimReady)
# ============================================================================

def estimate_physics_properties(
    asset: AssetSpec,
    glb_path: Path,
    client=None
) -> Dict[str, Any]:
    """
    Estimate physics properties for an asset.

    Uses category-based defaults, optionally refined by Gemini vision analysis.
    """
    category = asset.category.lower()
    defaults = PHYSICS_DEFAULTS.get(category, PHYSICS_DEFAULTS["default"])

    # Get mesh bounds if possible
    try:
        import trimesh
        mesh = trimesh.load(str(glb_path))
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        volume = float(size[0] * size[1] * size[2])
    except Exception:
        # Default small object size
        size = np.array([0.1, 0.1, 0.1])
        volume = 0.001

    # Calculate mass from density and volume
    mass = volume * defaults["bulk_density"]
    mass = max(0.01, min(mass, 100.0))  # Clamp to reasonable range

    # Build physics dict
    physics = {
        "mass_kg": float(mass),
        "bulk_density_kg_per_m3": float(defaults["bulk_density"]),
        "static_friction": float(defaults["static_friction"]),
        "dynamic_friction": float(defaults["dynamic_friction"]),
        "restitution": float(defaults["restitution"]),
        "collision_shape": str(defaults["collision_shape"]),
        "size_m": [float(s) for s in size],
        "volume_m3": float(volume),
        "center_of_mass_offset": [0.0, 0.0, 0.0],
        "semantic_class": asset.semantic_class,
        "graspable": True,
        "contact_offset_m": 0.005,
        "rest_offset_m": 0.001,
    }

    # Apply physics hints from manifest if provided
    if asset.physics_hints:
        hints = asset.physics_hints
        if "mass_range_kg" in hints:
            mass_range = hints["mass_range_kg"]
            # Use midpoint of range
            physics["mass_kg"] = float((mass_range[0] + mass_range[1]) / 2)
        if "friction" in hints:
            physics["static_friction"] = float(hints["friction"])
            physics["dynamic_friction"] = float(hints["friction"] * 0.85)
        if "collision_shape" in hints:
            physics["collision_shape"] = str(hints["collision_shape"])

    return physics


def write_simready_metadata(
    output_dir: Path,
    asset_name: str,
    asset: AssetSpec,
    physics: Dict[str, Any],
    source_info: Dict[str, Any]
) -> Path:
    """Write SimReady metadata JSON for the asset."""
    metadata = {
        "asset_name": asset_name,
        "category": asset.category,
        "semantic_class": asset.semantic_class,
        "description": asset.description,
        "physics": physics,
        "simready": True,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source": source_info,
    }

    meta_path = output_dir / safe_name(asset_name) / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


# ============================================================================
# Main Pipeline
# ============================================================================

def process_single_asset(
    asset: AssetSpec,
    config: PipelineConfig,
    client,
    output_dir: Path,
    vector_index: Optional[VectorIndex],
    scene_context: str = ""
) -> AssetResult:
    """
    Process a single asset through the complete pipeline.
    """
    timings = {}
    start_total = time.time()

    asset_safe_name = safe_name(asset.name)
    asset_out_dir = output_dir / asset_safe_name

    # Check if already exists
    usdz_path = asset_out_dir / "model.usdz"
    if config.skip_existing and usdz_path.is_file():
        print(f"[VAR-PIPELINE] Skipping existing: {asset.name}")
        return AssetResult(
            name=asset.name,
            success=True,
            stage_completed="skipped",
            output_path=str(usdz_path),
            asset_dir=str(asset_out_dir),
            reference_image=str(asset_out_dir / "reference.png"),
            metadata_path=str(asset_out_dir / "metadata.json"),
            timings={"total": 0.0},
        )

    # Check asset library
    if config.asset_library_path:
        library_asset = check_asset_library(
            asset,
            config.asset_library_path,
            vector_index,
            config,
            client,
        )
        if library_asset:
            output_path = copy_from_library(library_asset, output_dir, asset.name)
            return AssetResult(
                name=asset.name,
                success=True,
                stage_completed="library",
                output_path=str(output_path),
                asset_dir=str(asset_out_dir),
                reference_image=str(output_path.with_name(f"{output_path.stem}_thumb.png")),
                metadata_path=str(output_path.with_suffix('.json')),
                timings={"total": time.time() - start_total},
            )

    # Stage 1: Generate reference image
    start_image = time.time()
    success, image_path, error = generate_reference_image_gemini(
        client=client,
        asset=asset,
        output_dir=output_dir,
        scene_context=scene_context,
        dry_run=config.dry_run,
    )
    timings["image_generation"] = time.time() - start_image

    if not success:
        return AssetResult(
            name=asset.name,
            success=False,
            stage_completed="image",
            error=error,
            timings=timings,
        )

    if config.dry_run:
        backend, backend_selection_metadata = select_3d_backend(asset, config)
        return AssetResult(
            name=asset.name,
            success=True,
            stage_completed="complete",
            output_path=str(usdz_path),
            asset_dir=str(asset_out_dir),
            reference_image=str(image_path) if image_path else None,
            timings=timings,
            metadata={"backend_selection": backend_selection_metadata},
        )

    # Stage 2: Convert to 3D
    start_3d = time.time()
    backend, backend_selection_metadata = select_3d_backend(asset, config)

    preset = QUALITY_PRESETS.get(config.quality_mode, QUALITY_PRESETS["balanced"])

    if backend == "ultrashape":
        # UltraShape: Hunyuan + refinement for highest quality
        success, glb_path, error = run_ultrashape_reconstruction(
            image_path=image_path,
            output_dir=output_dir,
            asset_name=asset.name,
            render_size=preset["hunyuan_render_size"],
            texture_size=preset["hunyuan_texture_size"],
            ultrashape_steps=preset.get("ultrashape_steps", 50),
            ultrashape_octree_resolution=preset.get("ultrashape_octree_resolution", 256),
            dry_run=config.dry_run,
        )
    elif backend == "hunyuan":
        success, glb_path, error = run_hunyuan_reconstruction(
            image_path=image_path,
            output_dir=output_dir,
            asset_name=asset.name,
            render_size=preset["hunyuan_render_size"],
            texture_size=preset["hunyuan_texture_size"],
            dry_run=config.dry_run,
        )
    else:  # sam3d
        success, glb_path, error = run_sam3d_reconstruction(
            image_path=image_path,
            output_dir=output_dir,
            asset_name=asset.name,
            normalize=preset["sam3d_normalize"],
            dry_run=config.dry_run,
        )

    timings["3d_conversion"] = time.time() - start_3d
    timings["3d_backend"] = backend

    if not success:
        return AssetResult(
            name=asset.name,
            success=False,
            stage_completed="3d",
            error=error,
            timings=timings,
            metadata={"backend_selection": backend_selection_metadata},
        )

    # Stage 3: Add physics properties
    start_physics = time.time()
    physics = estimate_physics_properties(asset, glb_path, client)

    # Write metadata
    meta_path = write_simready_metadata(
        output_dir=output_dir,
        asset_name=asset.name,
        asset=asset,
        physics=physics,
        source_info={
            "pipeline": "variation-asset-pipeline",
            "3d_backend": backend,
            "quality_mode": config.quality_mode,
        },
    )
    timings["physics"] = time.time() - start_physics

    # Stage 4: Convert to USDZ
    start_usdz = time.time()
    usdz_success = convert_glb_to_usdz(glb_path, usdz_path)
    timings["usdz_conversion"] = time.time() - start_usdz

    if not usdz_success:
        # Fall back to GLB
        print(f"[VAR-PIPELINE] USDZ conversion failed, keeping GLB: {asset.name}")

    timings["total"] = time.time() - start_total

    return AssetResult(
        name=asset.name,
        success=True,
        stage_completed="complete",
        output_path=str(usdz_path if usdz_success else glb_path),
        timings=timings,
        metadata={
            "physics": physics,
            "backend_selection": backend_selection_metadata,
        },
        asset_dir=str(asset_out_dir),
        reference_image=str(image_path) if image_path else None,
        metadata_path=str(meta_path),
    )


def load_manifest(manifest_path: Path) -> Tuple[Dict[str, Any], List[AssetSpec]]:
    """Load and parse the variation assets manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    assets = []
    for asset_dict in manifest.get("assets", []):
        asset = AssetSpec(
            name=asset_dict.get("name", "unknown"),
            category=asset_dict.get("category", "other"),
            description=asset_dict.get("description", ""),
            semantic_class=asset_dict.get("semantic_class", "object"),
            priority=asset_dict.get("priority", "optional"),
            source_hint=asset_dict.get("source_hint"),
            example_variants=asset_dict.get("example_variants", []),
            physics_hints=asset_dict.get("physics_hints", {}),
            material_hint=asset_dict.get("material_hint"),
            style_hint=asset_dict.get("style_hint"),
            generation_prompt_hint=asset_dict.get("generation_prompt_hint"),
        )
        assets.append(asset)

    return manifest, assets


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run the complete variation asset pipeline."""

    print(f"[VAR-PIPELINE] Starting pipeline for scene: {config.scene_id}")
    print(f"[VAR-PIPELINE] 3D Backend: {config.backend_3d.value}")
    print(f"[VAR-PIPELINE] Quality Mode: {config.quality_mode}")

    # Load manifest
    manifest_path = GCS_ROOT / config.replicator_prefix / "variation_assets" / "manifest.json"

    try:
        manifest, assets = load_manifest(manifest_path)
    except FileNotFoundError as e:
        print(f"[VAR-PIPELINE] ERROR: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}

    print(f"[VAR-PIPELINE] Loaded manifest with {len(assets)} assets")
    print(f"[VAR-PIPELINE] Scene type: {manifest.get('scene_type', 'unknown')}")

    # Filter assets
    if config.priority_filter:
        assets = [a for a in assets if a.priority == config.priority_filter]
        print(f"[VAR-PIPELINE] Filtered to {len(assets)} {config.priority_filter} assets")

    # Filter to only assets that need generation
    assets_to_process = [
        a for a in assets
        if a.source_hint == "generate" or a.source_hint is None
    ]

    if config.max_assets:
        assets_to_process = assets_to_process[:config.max_assets]

    print(f"[VAR-PIPELINE] Processing {len(assets_to_process)} assets")

    if not assets_to_process:
        print("[VAR-PIPELINE] No assets to process")
        return {"success": True, "processed": 0, "results": []}

    # Create output directory
    output_dir = GCS_ROOT / config.variation_assets_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Gemini client
    if not config.dry_run:
        try:
            client = create_gemini_client()
        except ValueError as e:
            print(f"[VAR-PIPELINE] ERROR: {e}", file=sys.stderr)
            return {"success": False, "error": str(e)}
    else:
        client = None

    # Prepare vector index (optional)
    resolved_backend = resolve_vector_backend_uri(
        config.vector_backend_uri, config.asset_library_path
    )
    vector_index = load_vector_index(resolved_backend, config.asset_library_path)
    registry = AssetRegistry(
        backend_uri=config.registry_backend_uri,
        enabled=config.register_assets,
        dry_run=config.dry_run,
    )

    # Process assets
    scene_context = manifest.get("scene_type", "")
    results: List[AssetResult] = []

    for i, asset in enumerate(assets_to_process):
        print(f"[VAR-PIPELINE] Progress: {i + 1}/{len(assets_to_process)} - {asset.name}")

        result = process_single_asset(
            asset=asset,
            config=config,
            client=client,
            output_dir=output_dir,
            vector_index=vector_index,
            scene_context=scene_context,
        )
        results.append(result)

        if result.success:
            ingest_asset(
                asset=asset,
                result=result,
                config=config,
                registry=registry,
                source_pipeline="variation",
            )

        if result.success:
            print(f"[VAR-PIPELINE] ✓ {asset.name} ({result.stage_completed})")
        else:
            print(f"[VAR-PIPELINE] ✗ {asset.name}: {result.error}")

    # Generate summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    summary = {
        "success": failed == 0,
        "processed": len(results),
        "successful": successful,
        "failed": failed,
        "results": [asdict(r) for r in results],
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # Write summary
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Write assets manifest for downstream jobs
    assets_manifest = {
        "scene_id": config.scene_id,
        "generated_at": summary["generated_at"],
        "source": "variation-asset-pipeline",
        "assets": [],
    }

    for result in results:
        if result.success and result.output_path:
            assets_manifest["assets"].append({
                "name": result.name,
                "path": result.output_path,
                "simready": True,
                "metadata": result.metadata,
            })

    assets_manifest_path = output_dir / "simready_assets.json"
    with assets_manifest_path.open("w") as f:
        json.dump(assets_manifest, f, indent=2)

    # Write completion marker
    marker_path = output_dir / ".variation_pipeline_complete"
    marker_path.write_text(
        f"completed at {summary['generated_at']}\n"
        f"successful: {successful}\n"
        f"failed: {failed}\n"
    )

    print(f"[VAR-PIPELINE] Pipeline complete!")
    print(f"[VAR-PIPELINE]   Successful: {successful}")
    print(f"[VAR-PIPELINE]   Failed: {failed}")
    print(f"[VAR-PIPELINE]   Output: {output_dir}")

    return summary


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""

    # Parse configuration from environment
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[VAR-PIPELINE]",
    )

    scene_id = os.environ["SCENE_ID"]
    bucket = os.environ["BUCKET"]

    dry_run = getenv_bool("DRY_RUN", "0")
    register_assets = getenv_bool("REGISTER_ASSETS", "1") and not dry_run

    config = PipelineConfig(
        scene_id=scene_id,
        bucket=bucket or None,
        replicator_prefix=os.getenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator"),
        variation_assets_prefix=os.getenv("VARIATION_ASSETS_PREFIX", f"scenes/{scene_id}/variation_assets"),
        backend_3d=Backend3D(os.getenv("3D_BACKEND", "auto").lower()),
        quality_mode=os.getenv("QUALITY_MODE", "balanced"),
        max_assets=int(os.getenv("MAX_ASSETS")) if os.getenv("MAX_ASSETS") else None,
        priority_filter=os.getenv("PRIORITY_FILTER") or None,
        asset_library_path=os.getenv("ASSET_LIBRARY_PATH"),
        registry_backend_uri=os.getenv("ASSET_REGISTRY_BACKEND_URI"),
        register_assets=register_assets,
        skip_existing=getenv_bool("SKIP_EXISTING", "1"),
        dry_run=dry_run,
        vector_backend_uri=os.getenv("ASSET_VECTOR_BACKEND_URI"),
        vector_similarity_threshold=float(
            os.getenv("ASSET_VECTOR_SIMILARITY_THRESHOLD", "0.82")
        ),
        vector_max_candidates=int(os.getenv("ASSET_VECTOR_MAX_CANDIDATES", "5")),
    )

    print(f"[VAR-PIPELINE] Configuration:")
    print(f"[VAR-PIPELINE]   Scene ID: {config.scene_id}")
    print(f"[VAR-PIPELINE]   Bucket: {bucket}")
    print(f"[VAR-PIPELINE]   Replicator prefix: {config.replicator_prefix}")
    print(f"[VAR-PIPELINE]   Output prefix: {config.variation_assets_prefix}")
    print(f"[VAR-PIPELINE]   3D Backend: {config.backend_3d.value}")
    print(f"[VAR-PIPELINE]   Quality Mode: {config.quality_mode}")
    if config.max_assets:
        print(f"[VAR-PIPELINE]   Max Assets: {config.max_assets}")
    if config.priority_filter:
        print(f"[VAR-PIPELINE]   Priority Filter: {config.priority_filter}")
    if config.asset_library_path:
        print(f"[VAR-PIPELINE]   Asset Library: {config.asset_library_path}")
    if config.registry_backend_uri:
        print(f"[VAR-PIPELINE]   Registry backend: {config.registry_backend_uri}")
    print(
        f"[VAR-PIPELINE]   Registry updates: {'enabled' if config.register_assets else 'disabled'}"
    )
    if config.vector_backend_uri:
        print(f"[VAR-PIPELINE]   Vector backend: {config.vector_backend_uri}")
    print(
        f"[VAR-PIPELINE]   Vector similarity threshold: {config.vector_similarity_threshold}"
    )
    print(
        f"[VAR-PIPELINE]   Vector max candidates: {config.vector_max_candidates}"
    )
    if config.dry_run:
        print(f"[VAR-PIPELINE]   DRY RUN MODE")

    try:
        result = run_pipeline(config)

        if result.get("success"):
            print("[VAR-PIPELINE] SUCCESS")
            sys.exit(0)
        else:
            print(f"[VAR-PIPELINE] COMPLETED WITH ERRORS")
            sys.exit(0)  # Don't fail for partial success

    except Exception as e:
        print(f"[VAR-PIPELINE] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="VARIATION-ASSET-PIPELINE", validate_gcs=True)
    main()
