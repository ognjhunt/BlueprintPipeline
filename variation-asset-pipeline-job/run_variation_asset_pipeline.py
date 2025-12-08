#!/usr/bin/env python3
"""
Variation Asset Pipeline - End-to-End SimReady Asset Generation

This job orchestrates the complete pipeline for generating variation assets
needed for domain randomization in Isaac Sim Replicator:

1. Read manifest from replicator-job (variation_assets/manifest.json)
2. Check asset library for existing assets (optional optimization)
3. For missing assets:
   a. Generate reference images using Gemini 2.0 Flash
   b. Convert to 3D using SAM3D (fast) or Hunyuan (quality)
   c. Add physics properties inline (mass, friction, COM, collision)
4. Output: SimReady USDZ assets ready for Replicator placement

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

    3D_BACKEND: "sam3d" | "hunyuan" | "auto" (default: "auto")
    QUALITY_MODE: "fast" | "balanced" | "quality" (default: "balanced")
    MAX_ASSETS: Maximum number of assets to generate (default: all)
    PRIORITY_FILTER: "required" | "recommended" | "optional" | "" (default: "")

    ASSET_LIBRARY_PATH: Path to shared asset library (optional)
    SKIP_EXISTING: Skip assets that already exist (default: "1")
    DRY_RUN: Skip actual generation (default: "0")
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
import io
import base64

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

# Gemini model for image generation (Gemini 2.0 Flash with native image gen)
GEMINI_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"

# Quality presets
QUALITY_PRESETS = {
    "fast": {
        "3d_backend": "sam3d",
        "image_size": 512,
        "sam3d_normalize": True,
        "hunyuan_render_size": 512,
        "hunyuan_texture_size": 1024,
    },
    "balanced": {
        "3d_backend": "auto",  # SAM3D for simple, Hunyuan for complex
        "image_size": 1024,
        "sam3d_normalize": True,
        "hunyuan_render_size": 1024,
        "hunyuan_texture_size": 2048,
    },
    "quality": {
        "3d_backend": "hunyuan",
        "image_size": 2048,
        "sam3d_normalize": True,
        "hunyuan_render_size": 2048,
        "hunyuan_texture_size": 4096,
    },
}

# Category-based complexity hints for auto backend selection
SIMPLE_CATEGORIES = {"dishes", "utensils", "cans", "bottles", "boxes", "containers"}
COMPLEX_CATEGORIES = {"clothing", "food", "produce", "tools", "electronics", "lab_equipment"}

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
    error: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    scene_id: str
    replicator_prefix: str
    variation_assets_prefix: str
    backend_3d: Backend3D
    quality_mode: str
    max_assets: Optional[int]
    priority_filter: Optional[str]
    asset_library_path: Optional[str]
    skip_existing: bool
    dry_run: bool


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


# ============================================================================
# Asset Library (Optional)
# ============================================================================

def check_asset_library(
    asset: AssetSpec,
    library_path: Optional[str]
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

    # Could also check by semantic similarity here in the future
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

    return output_path


# ============================================================================
# Image Generation (Gemini 2.0 Flash)
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
    Generate a reference image using Gemini 2.0 Flash native image generation.

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
            # Use Gemini 2.0 Flash with native image generation
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

def select_3d_backend(asset: AssetSpec, config: PipelineConfig) -> str:
    """
    Select 3D backend based on asset complexity and config.
    """
    if config.backend_3d != Backend3D.AUTO:
        return config.backend_3d.value

    # Auto selection based on category
    category_lower = asset.category.lower()

    if category_lower in SIMPLE_CATEGORIES:
        return "sam3d"
    elif category_lower in COMPLEX_CATEGORIES:
        return "hunyuan"
    else:
        # Default to SAM3D for speed
        return "sam3d"


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
    In production, this would call the sam3d-job or use the Inference class.

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


def convert_glb_to_usdz(glb_path: Path, usdz_path: Path) -> bool:
    """Convert GLB to USDZ using usd_from_gltf if available."""
    usd_from_gltf = shutil.which("usd_from_gltf")
    if usd_from_gltf is None:
        print("[VAR-PIPELINE] WARNING: usd_from_gltf not available", file=sys.stderr)
        return False

    usdz_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [usd_from_gltf, str(glb_path), "-o", str(usdz_path)],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[VAR-PIPELINE] usd_from_gltf failed: {e.stderr}", file=sys.stderr)
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
            timings={"total": 0.0},
        )

    # Check asset library
    if config.asset_library_path:
        library_asset = check_asset_library(asset, config.asset_library_path)
        if library_asset:
            output_path = copy_from_library(library_asset, output_dir, asset.name)
            return AssetResult(
                name=asset.name,
                success=True,
                stage_completed="library",
                output_path=str(output_path),
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
        return AssetResult(
            name=asset.name,
            success=True,
            stage_completed="complete",
            output_path=str(usdz_path),
            timings=timings,
        )

    # Stage 2: Convert to 3D
    start_3d = time.time()
    backend = select_3d_backend(asset, config)

    preset = QUALITY_PRESETS.get(config.quality_mode, QUALITY_PRESETS["balanced"])

    if backend == "hunyuan":
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
        metadata={"physics": physics},
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
            scene_context=scene_context,
        )
        results.append(result)

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
    scene_id = os.getenv("SCENE_ID", "")
    bucket = os.getenv("BUCKET", "")

    if not scene_id:
        print("[VAR-PIPELINE] ERROR: SCENE_ID is required", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        scene_id=scene_id,
        replicator_prefix=os.getenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator"),
        variation_assets_prefix=os.getenv("VARIATION_ASSETS_PREFIX", f"scenes/{scene_id}/variation_assets"),
        backend_3d=Backend3D(os.getenv("3D_BACKEND", "auto").lower()),
        quality_mode=os.getenv("QUALITY_MODE", "balanced"),
        max_assets=int(os.getenv("MAX_ASSETS")) if os.getenv("MAX_ASSETS") else None,
        priority_filter=os.getenv("PRIORITY_FILTER") or None,
        asset_library_path=os.getenv("ASSET_LIBRARY_PATH"),
        skip_existing=getenv_bool("SKIP_EXISTING", "1"),
        dry_run=getenv_bool("DRY_RUN", "0"),
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
    main()
