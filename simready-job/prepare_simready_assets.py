import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.scene_manifest.loader import load_manifest_or_scene_assets

import numpy as np
from PIL import Image  # type: ignore
from tools.asset_catalog import AssetCatalogClient

try:
    # Google GenAI SDK for Gemini 3.x
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

# Unified LLM client supporting Gemini + OpenAI
try:
    from tools.llm_client import create_llm_client, LLMProvider, LLMResponse
    HAVE_LLM_CLIENT = True
except ImportError:  # pragma: no cover
    HAVE_LLM_CLIENT = False
    create_llm_client = None
    LLMProvider = None
    LLMResponse = None

GCS_ROOT = Path("/mnt/gcs")


# ---------- Small helpers ----------

def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_path_join(root: Path, rel: str) -> Path:
    rel_path = rel.lstrip("/")
    return root / rel_path


def choose_reference_image_path(root: Path, obj: Dict[str, Any]) -> Optional[Path]:
    """
    Pick the best available reference image for an object.

    Preference (same as other jobs): preferred_view -> multiview/view_0.png -> crop.
    """

    candidates: List[Path] = []

    preferred_rel = obj.get("preferred_view")
    mv_rel = obj.get("multiview_dir")
    crop_rel = obj.get("crop_path")

    if isinstance(preferred_rel, str):
        candidates.append(safe_path_join(root, preferred_rel))
    if isinstance(mv_rel, str):
        candidates.append(safe_path_join(root, mv_rel) / "view_0.png")
    if isinstance(crop_rel, str):
        candidates.append(safe_path_join(root, crop_rel))

    for p in candidates:
        if p.is_file():
            return p

    return None


def load_image_for_gemini(image_path: Path) -> Optional["Image.Image"]:
    """
    Load an RGB PIL image for Gemini input.
    """

    try:
        return Image.open(str(image_path)).convert("RGB")
    except Exception as exc:  # pragma: no cover - image decoding errors
        print(f"[SIMREADY] WARNING: failed to load reference image {image_path}: {exc}", file=sys.stderr)
        return None


def _as_float_list(v: Any, n: int) -> Optional[List[float]]:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    out: List[float] = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            return None
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def load_object_metadata(
    root: Path,
    obj: Dict[str, Any],
    assets_prefix: str,
    catalog_client: Optional[AssetCatalogClient] = None,
) -> Optional[dict]:
    """
    Same lookup strategy as usd-assembly build_scene_usd:
    1. catalog lookup by asset_id or asset_path
    2. explicit metadata_path
    3. next to asset_path
    4. fallback: assets/static/obj_{id}/metadata.json
    """
    if catalog_client is not None:
        logical_id = obj.get("logical_asset_id") or obj.get("logical_id")
        try:
            catalog_meta = catalog_client.lookup_metadata(
                asset_id=obj.get("id"),
                asset_path=obj.get("asset_path"),
                logical_id=logical_id,
            )
            if catalog_meta:
                return catalog_meta
        except Exception as exc:  # pragma: no cover - network errors
            print(f"[SIMREADY] WARNING: catalog lookup failed: {exc}", file=sys.stderr)

    metadata_rel = obj.get("metadata_path")
    if metadata_rel:
        candidate = safe_path_join(root, metadata_rel)
        if candidate.is_file():
            return json.loads(candidate.read_text())

    asset_path = obj.get("asset_path")
    if asset_path:
        asset_dir = safe_path_join(root, asset_path).parent
        candidate = asset_dir / "metadata.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())

    oid = obj.get("id")
    if oid is not None:
        static_dir = safe_path_join(root, f"{assets_prefix}/static/obj_{oid}")
        candidate = static_dir / "metadata.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())

    return None


def extract_mesh_bounds_from_metadata(metadata: Optional[dict]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Try to read an axis-aligned bounding box from metadata.
    Returns: (size_xyz_m, center_xyz_m)
    """
    if not metadata:
        return None, None

    mesh_bounds = metadata.get("mesh_bounds") or {}
    export_bounds = mesh_bounds.get("export") or mesh_bounds.get("bounds") or mesh_bounds

    size = _as_float_list(export_bounds.get("size"), 3)

    center = _as_float_list(export_bounds.get("center"), 3)

    bmin = _as_float_list(export_bounds.get("min") or export_bounds.get("minimum"), 3)
    bmax = _as_float_list(export_bounds.get("max") or export_bounds.get("maximum"), 3)

    if size is None and bmin is not None and bmax is not None:
        size = [float(bmax[i] - bmin[i]) for i in range(3)]

    if center is None and bmin is not None and bmax is not None:
        center = [float((bmin[i] + bmax[i]) * 0.5) for i in range(3)]

    # Sanitize obvious nonsense
    if size is not None:
        size = [max(float(s), 1e-3) for s in size]

    return size, center


def extract_obb_bounds_from_obj(obj: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Best-effort OBB fallback.
    Many pipelines store OBB "extents" as half-extents. We assume half-extents and multiply by 2.
    Returns: (size_xyz_m, center_xyz_m)
    """
    obb = obj.get("obb") or {}
    extents = _as_float_list(obb.get("extents") or obb.get("half_extents"), 3)
    center = _as_float_list(obb.get("center"), 3)

    size = None
    if extents is not None:
        size = [max(2.0 * float(e), 1e-3) for e in extents]

    return size, center


def compute_bounds(
    obj: Dict[str, Any],
    metadata: Optional[dict],
    gemini_estimated_size: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute a usable axis-aligned bounds estimate in meters:
      - prefer metadata mesh_bounds
      - fallback to obj.obb
      - fallback to gemini_estimated_size (if provided)
      - else default 10cm cube
    """
    size, center = extract_mesh_bounds_from_metadata(metadata)
    source = "metadata"

    if size is None:
        size2, center2 = extract_obb_bounds_from_obj(obj)
        if size2 is not None:
            size, center = size2, center2
            source = "obb"

    if size is None and gemini_estimated_size is not None:
        size = gemini_estimated_size
        source = "gemini_estimated"

    if size is None:
        size = [0.10, 0.10, 0.10]  # 10cm cube fallback
        source = "default"

    if center is None:
        center = [0.0, 0.0, 0.0]

    sx, sy, sz = [max(float(s), 1e-3) for s in size]
    volume = float(sx * sy * sz)

    return {
        "size_m": [sx, sy, sz],
        "center_m": [float(center[0]), float(center[1]), float(center[2])],
        "volume_m3": volume,
        "source": source,
    }


def padded_size(size_m: List[float]) -> List[float]:
    """
    Apply a small collision padding so fast contacts are less likely to tunnel.
    Tunable via env vars:
      SIMREADY_COLLIDER_PAD_RATIO (default 0.01 == 1%)
      SIMREADY_COLLIDER_PAD_MIN_M (default 0.001 == 1mm)
      SIMREADY_COLLIDER_PAD_MAX_M (default 0.02 == 2cm)
    """
    ratio = float(os.getenv("SIMREADY_COLLIDER_PAD_RATIO", "0.01"))
    pad_min = float(os.getenv("SIMREADY_COLLIDER_PAD_MIN_M", "0.001"))
    pad_max = float(os.getenv("SIMREADY_COLLIDER_PAD_MAX_M", "0.02"))

    mx = max(size_m)
    pad = _clamp(mx * ratio, pad_min, pad_max)
    return [max(float(s) + 2.0 * pad, 1e-3) for s in size_m]


# ---------- Generic fallback (used only when Gemini fails) ----------

GENERIC_FALLBACK: Dict[str, Any] = {
    # Effective bulk density (includes air gaps); "generic solid-ish household thing"
    "bulk_density_kg_per_m3": 600.0,
    # Very wide range: 1g to 1000kg
    "mass_range_kg": (0.001, 1000.0),
    "static_friction": 0.6,
    "dynamic_friction": 0.5,
    "restitution": 0.1,
    "dynamic": True,
    "collision_shape": "box",  # box/sphere/capsule
    "material_name": "generic",
    # New robotics-focused properties
    "semantic_class": "object",
    "center_of_mass_offset": [0.0, 0.0, 0.0],  # offset from geometric center
    "graspable": True,
    "grasp_regions": [],  # list of {"position": [x,y,z], "type": "pinch|power|wrap", "width_m": float}
    "contact_offset_m": 0.005,  # PhysX contact detection margin
    "rest_offset_m": 0.001,  # PhysX rest separation
    "surface_roughness": 0.5,  # 0=smooth, 1=rough (for tactile simulation)
}


# ---------- Sim2Real Distribution Parameters ----------
# These define uncertainty ranges for physics properties to enable
# domain randomization during training, improving sim2real transfer.

def compute_physics_distributions(physics: Dict[str, Any], bounds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute distribution ranges for physics properties to enable domain randomization.

    Returns a dict with *_range fields for key physics properties:
    - mass_kg_range: [min, max] for mass randomization
    - friction_static_range: [min, max] for static friction
    - friction_dynamic_range: [min, max] for dynamic friction
    - restitution_range: [min, max] for bounciness
    - center_of_mass_noise: [x, y, z] noise magnitude per axis
    - contact_offset_range: [min, max] for PhysX contact offset

    These ranges are designed for realistic domain randomization that
    improves policy robustness without creating implausible physics.
    """
    distributions: Dict[str, Any] = {}

    # Mass distribution: ±20% of estimated mass (common manufacturing variance)
    mass = float(physics.get("mass_kg", 1.0))
    mass_variance = 0.20
    distributions["mass_kg_range"] = [
        max(0.001, mass * (1.0 - mass_variance)),
        mass * (1.0 + mass_variance)
    ]

    # Friction distributions: ±15% variance (surface condition variance)
    static_f = float(physics.get("static_friction", 0.6))
    dynamic_f = float(physics.get("dynamic_friction", 0.5))
    friction_variance = 0.15

    distributions["friction_static_range"] = [
        max(0.1, static_f * (1.0 - friction_variance)),
        min(1.5, static_f * (1.0 + friction_variance))
    ]
    distributions["friction_dynamic_range"] = [
        max(0.05, dynamic_f * (1.0 - friction_variance)),
        min(1.2, dynamic_f * (1.0 + friction_variance))
    ]

    # Restitution distribution: ±30% (material property variance)
    restitution = float(physics.get("restitution", 0.1))
    restitution_variance = 0.30
    distributions["restitution_range"] = [
        max(0.0, restitution * (1.0 - restitution_variance)),
        min(1.0, restitution * (1.0 + restitution_variance))
    ]

    # Center of mass noise: proportional to object size (5% of each dimension)
    size_m = list(bounds.get("size_m") or [0.1, 0.1, 0.1])
    com_noise_ratio = 0.05
    distributions["center_of_mass_noise"] = [
        size_m[0] * com_noise_ratio,
        size_m[1] * com_noise_ratio,
        size_m[2] * com_noise_ratio
    ]

    # Contact offset range: ±50% (solver sensitivity variance)
    contact_offset = float(physics.get("contact_offset_m", 0.005))
    distributions["contact_offset_range"] = [
        max(0.001, contact_offset * 0.5),
        min(0.02, contact_offset * 1.5)
    ]

    # Surface roughness range: ±20%
    roughness = float(physics.get("surface_roughness", 0.5))
    roughness_variance = 0.20
    distributions["surface_roughness_range"] = [
        max(0.0, roughness * (1.0 - roughness_variance)),
        min(1.0, roughness * (1.0 + roughness_variance))
    ]

    return distributions


# ---------- Physics config (default + Gemini) ----------

def estimate_default_physics(obj: Dict[str, Any], bounds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baseline physics estimate used only when Gemini is unavailable or fails.
    """
    volume = float(bounds.get("volume_m3") or 0.0)
    rho = float(GENERIC_FALLBACK["bulk_density_kg_per_m3"])

    if volume > 0.0:
        mass_est = volume * rho
        mass = float(np.clip(mass_est, GENERIC_FALLBACK["mass_range_kg"][0], GENERIC_FALLBACK["mass_range_kg"][1]))
    else:
        mass = 1.0

    notes = f"fallback estimate; bulk_density={rho:.1f} kg/m^3; volume={volume:.6f} m^3; bounds_source={bounds.get('source')}"

    return {
        "dynamic": bool(GENERIC_FALLBACK["dynamic"]),
        "collision_shape": str(GENERIC_FALLBACK["collision_shape"]),
        "mass_kg": float(mass),
        "bulk_density_kg_per_m3": float(rho),
        "static_friction": float(GENERIC_FALLBACK["static_friction"]),
        "dynamic_friction": float(GENERIC_FALLBACK["dynamic_friction"]),
        "restitution": float(GENERIC_FALLBACK["restitution"]),
        "material_name": str(GENERIC_FALLBACK["material_name"]),
        "notes": notes,
        # Robotics-focused properties
        "semantic_class": str(GENERIC_FALLBACK["semantic_class"]),
        "center_of_mass_offset": list(GENERIC_FALLBACK["center_of_mass_offset"]),
        "graspable": bool(GENERIC_FALLBACK["graspable"]),
        "grasp_regions": list(GENERIC_FALLBACK["grasp_regions"]),
        "contact_offset_m": float(GENERIC_FALLBACK["contact_offset_m"]),
        "rest_offset_m": float(GENERIC_FALLBACK["rest_offset_m"]),
        "surface_roughness": float(GENERIC_FALLBACK["surface_roughness"]),
    }


def _compact_obj_description(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep prompt context rich but bounded (avoid dumping huge dicts).
    """
    keep: Dict[str, Any] = {}
    priority_keys = [
        "id",
        "name",
        "label",
        "type",
        "class_name",
        "category",
        "pipeline",
        "asset_path",
        "metadata_path",
        "source",
        "tags",
        "description",
        "prompt",
        "caption",
    ]

    for k in priority_keys:
        if k in obj:
            v = obj.get(k)
            if isinstance(v, (str, int, float, bool)):
                keep[k] = v
            elif isinstance(v, list) and len(v) <= 20:
                # keep short lists of primitives/strings
                if all(isinstance(x, (str, int, float, bool)) for x in v):
                    keep[k] = v

    # Add a small sample of other short strings
    for k, v in obj.items():
        if k in keep or k in {"obb"}:
            continue
        if isinstance(v, str) and 0 < len(v) <= 120:
            keep[k] = v
            if len(keep) >= 20:
                break

    # Keep OBB hint, but not huge matrices
    obb = obj.get("obb")
    if isinstance(obb, dict):
        for kk in ("extents", "center"):
            if kk in obb:
                vv = obb.get(kk)
                if isinstance(vv, (list, tuple)) and len(vv) == 3 and all(isinstance(x, (int, float)) for x in vv):
                    keep[f"obb_{kk}"] = [float(x) for x in vv]

    return keep


def make_gemini_prompt(
    oid: Any,
    obj: Dict[str, Any],
    bounds: Dict[str, Any],
    base_cfg: Dict[str, Any],
    has_image: bool = False,
) -> str:
    """
    Prompt Gemini to estimate realistic physics parameters based on object metadata + size.
    """
    size = bounds.get("size_m")
    center = bounds.get("center_m")
    volume = bounds.get("volume_m3")

    obj_desc = _compact_obj_description(obj)
    obj_desc["id"] = oid

    if has_image:
        image_context = (
            "You must estimate REALISTIC physics parameters for the *specific object* shown in the attached image,"
            " using its appearance as the primary signal and the text metadata below as context."
        )
        primary_guidance = (
            "You are given a reference image of the object. Base your estimates primarily on that photo (shape, material, heft), "
            "and use the metadata below only as supporting context."
        )
    else:
        image_context = (
            "You must estimate REALISTIC physics parameters for the object using the metadata and scale information below."
        )
        primary_guidance = (
            "No photo was provided. Rely on the metadata below plus the size to estimate realistic properties, "
            "but avoid generic placeholders."
        )

    skeleton = {
        "dynamic": True,
        "collision_shape": "box",  # box|sphere|capsule
        "mass_kg": 1.0,  # total mass of this object at this scale
        "bulk_density_kg_per_m3": 600.0,  # effective density incl. voids; should be consistent with mass_kg/volume
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "restitution": 0.1,
        "material_name": "generic",
        "notes": "",
        # Robotics-focused properties
        "semantic_class": "object",  # e.g. container, tool, food, electronics, furniture, toy, clothing, book, kitchenware
        "center_of_mass_offset": [0.0, 0.0, 0.0],  # offset from geometric center in meters [x, y, z]
        "graspable": True,  # can a robot gripper pick this up?
        "grasp_regions": [],  # list of grasp points: [{"position": [x,y,z], "type": "pinch|power|wrap", "width_m": 0.05}]
        "surface_roughness": 0.5,  # 0.0=glass-smooth, 1.0=sandpaper-rough (affects tactile sensing)
    }

    prompt = f"""
You are configuring physics for USD assets to be simulated in NVIDIA Isaac Sim (PhysX / USD Physics).
These assets will be used by ROBOTICS COMPANIES for manipulation training, so accuracy matters.

{image_context}

{primary_guidance}

Important implementation notes (affect what "realistic" means here):
- The sim uses meters and kilograms.
- We will generate an *analytic collision proxy* (box/sphere/capsule) from the object's bounds.
- In USD Physics, mass and density can both exist, but *explicit mass overrides density*.
  Provide values that are consistent with the provided volume (when available).

### Object bounds (meters)
- size_m (AABB): {json.dumps(size)}
- center_m (AABB center): {json.dumps(center)}
- approx_volume_m3: {json.dumps(volume)}
- bounds_source: {json.dumps(bounds.get("source"))}

### Guidance (use as reality checks, not as constraints)
- Typical solid material densities (kg/m^3): plastics ~900-1400, wood ~400-900, glass/ceramic ~2000-3000,
  aluminum ~2700, steel ~7800. Note: many consumer objects are hollow, so effective/bulk density can be much lower.
- Static friction is usually >= dynamic friction. Most household objects have low restitution (~0.0-0.2).
- Collision shape: choose "box" for blocky/general shapes, "sphere" for near-spherical, "capsule" for elongated.

### ROBOTICS-SPECIFIC GUIDANCE (critical for manipulation tasks):

1. **semantic_class**: Choose the most specific applicable category:
   - container, tool, food, electronics, furniture, toy, clothing, book, kitchenware,
   - bottle, can, box, bag, utensil, appliance, decoration, plant, sporting_goods, office_supply

2. **center_of_mass_offset**: Offset from geometric center in meters [x, y, z].
   - For uniform objects: [0, 0, 0]
   - For objects heavier at bottom (e.g., bottles, vases): negative Y offset like [0, -0.02, 0]
   - For objects with asymmetric mass (e.g., hammer): offset toward heavy end
   - Consider: filled containers have CoM based on contents, not shell

3. **graspable**: Can a parallel-jaw or suction gripper pick this up?
   - False for: very large furniture, fixtures, things bolted down
   - True for: most household objects under ~20kg

4. **grasp_regions**: Suggest 1-3 good grasp locations relative to object center.
   Each region: {{"position": [x, y, z], "type": "pinch|power|wrap", "width_m": <gripper_opening>}}
   - pinch: precision grip with fingertips (small objects, edges)
   - power: full-hand cylindrical grip (handles, bottles)
   - wrap: enveloping grip (irregular shapes)
   - width_m: required gripper opening in meters (0.02-0.15 typical)
   - For simple objects, one central grasp is fine. For tools/handles, suggest multiple.

5. **surface_roughness**: 0.0 (glass-smooth) to 1.0 (sandpaper-rough)
   - Polished metal/glass: 0.1-0.2
   - Painted/coated plastic: 0.3-0.4
   - Bare wood: 0.5-0.6
   - Textured rubber/fabric: 0.7-0.9

CRITICAL:
- Do NOT return generic placeholder values.
- Make mass_kg plausible for this object *at this scale*.
- If you output both mass_kg and bulk_density_kg_per_m3, they should approximately agree with:
    mass_kg ≈ bulk_density_kg_per_m3 * approx_volume_m3
  (within a few times is fine because bounds include air/voids).
- For grasp_regions, be SPECIFIC about where on this particular object a robot should grip.

Return ONLY valid JSON (no markdown, no comments, no extra text) matching this structure:

{json.dumps(skeleton, indent=2)}

Object info (cropped to relevant fields):

{json.dumps(obj_desc, indent=2)}
"""
    return prompt


def have_gemini() -> bool:
    """Check if Gemini is available. Supports both GEMINI_API_KEY and GOOGLE_API_KEY."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai is not None and types is not None and bool(api_key)


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment. Checks both GEMINI_API_KEY and GOOGLE_API_KEY."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def have_openai() -> bool:
    """Check if OpenAI is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def have_any_llm() -> bool:
    """Check if any LLM provider is available."""
    return have_gemini() or have_openai()


def get_llm_provider() -> Optional[str]:
    """Get the preferred LLM provider based on environment."""
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider == "openai" and have_openai():
        return "openai"
    elif provider == "gemini" and have_gemini():
        return "gemini"
    elif provider == "auto":
        if have_gemini():
            return "gemini"
        elif have_openai():
            return "openai"
    return None


def load_multiview_images_for_gemini(
    root: Path, obj: Dict[str, Any], max_views: int = 4
) -> List["Image.Image"]:
    """
    Load multiple view images from the multiview directory for better dimension estimation.
    Returns a list of PIL images (up to max_views).
    """
    images: List["Image.Image"] = []
    mv_rel = obj.get("multiview_dir")
    if not isinstance(mv_rel, str):
        return images

    mv_dir = safe_path_join(root, mv_rel)
    if not mv_dir.is_dir():
        return images

    # Try to load views 0-3 (front, side, back, top typically)
    for i in range(max_views):
        view_path = mv_dir / f"view_{i}.png"
        if view_path.is_file():
            img = load_image_for_gemini(view_path)
            if img is not None:
                images.append(img)

    return images


def make_dimension_estimation_prompt(obj: Dict[str, Any], has_multiple_views: bool = False) -> str:
    """
    Create a prompt for Gemini to estimate real-world dimensions of an object from images.
    """
    obj_desc = _compact_obj_description(obj)

    if has_multiple_views:
        image_context = (
            "You are provided with MULTIPLE VIEWS of the same 3D object. "
            "Use all views together to estimate the real-world dimensions more accurately. "
            "The views typically include front, side, and other angles."
        )
    else:
        image_context = (
            "You are provided with a single image of a 3D object. "
            "Estimate its real-world dimensions based on visual appearance."
        )

    prompt = f"""
You are an expert at estimating real-world dimensions of objects from images.

{image_context}

Your task is to estimate the REAL-WORLD physical dimensions (width, height, depth) of the object in METERS.

Important guidelines:
- Think about what this object actually is and what its typical real-world size would be.
- Consider common reference objects: a book is typically 0.15-0.30m tall, a chair seat is ~0.45m high,
  a coffee mug is ~0.10m tall, a dining table is ~0.75m high and 1.0-2.0m wide.
- For objects like furniture, appliances, or decor items, use your knowledge of typical real-world sizes.
- The dimensions should represent the axis-aligned bounding box of the object.

Object metadata (for context):
{json.dumps(obj_desc, indent=2)}

Return ONLY valid JSON (no markdown, no comments, no extra text) with this exact structure:
{{
    "width_m": <float>,
    "height_m": <float>,
    "depth_m": <float>,
    "confidence": "<low|medium|high>",
    "reasoning": "<brief explanation of how you determined the size>"
}}

Where:
- width_m: extent along X axis (left-right)
- height_m: extent along Y axis (up-down)
- depth_m: extent along Z axis (front-back)
- All values in meters (e.g., 0.30 for 30cm)
"""
    return prompt


def estimate_scale_gemini(
    client: "genai.Client",
    obj: Dict[str, Any],
    reference_image: "Image.Image",
    class_name: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Estimate real-world scale of an object using Gemini vision.

    This is a convenience wrapper around call_gemini_for_dimensions that
    works with a single reference image.

    Args:
        client: Gemini API client
        obj: Object dict with metadata
        reference_image: PIL Image of the object
        class_name: Optional class name override

    Returns:
        [width, height, depth] in meters, or None if estimation fails
    """
    if client is None or reference_image is None:
        return None

    # Add class_name to obj if provided
    obj_copy = dict(obj)
    if class_name:
        obj_copy["class_name"] = class_name

    return call_gemini_for_dimensions(client, obj_copy, [reference_image])


def call_gemini_for_dimensions(
    client: "genai.Client",
    obj: Dict[str, Any],
    images: List["Image.Image"],
) -> Optional[List[float]]:
    """
    Ask Gemini to estimate real-world dimensions of an object from images.
    Returns [width, height, depth] in meters, or None if estimation fails.
    """
    if client is None or not images:
        return None

    prompt = make_dimension_estimation_prompt(obj, has_multiple_views=len(images) > 1)

    try:
        # Use the latest Gemini model for best physics estimation
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-06-05")

        cfg_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
        }

        # Enable grounding for supported Gemini models (2.5+)
        grounding_enabled = os.getenv("GEMINI_GROUNDING_ENABLED", "false").lower() in {"1", "true", "yes"}
        if grounding_enabled:
            if hasattr(types, "Tool") and hasattr(types, "GoogleSearch"):
                try:
                    cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                except Exception:
                    pass  # Grounding not supported for this model

        try:
            config = types.GenerateContentConfig(**cfg_kwargs)
        except Exception:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )

        # Build content: images first, then prompt
        contents = list(images) + [prompt]

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        raw = _strip_code_fences(response.text or "")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")

        width = float(data.get("width_m", 0))
        height = float(data.get("height_m", 0))
        depth = float(data.get("depth_m", 0))
        confidence = data.get("confidence", "low")
        reasoning = data.get("reasoning", "")

        # Sanity checks: dimensions should be reasonable (1mm to 10m)
        if not (0.001 <= width <= 10.0 and 0.001 <= height <= 10.0 and 0.001 <= depth <= 10.0):
            print(f"[SIMREADY] WARNING: Gemini dimension estimate out of range: {width}x{height}x{depth}m", file=sys.stderr)
            return None

        print(f"[SIMREADY] Gemini estimated dimensions: {width:.3f}x{height:.3f}x{depth:.3f}m (confidence: {confidence})")
        if reasoning:
            print(f"[SIMREADY]   Reasoning: {reasoning[:200]}")

        return [width, height, depth]

    except Exception as e:
        print(f"[SIMREADY] WARNING: Gemini dimension estimation failed: {e}", file=sys.stderr)
        return None


def _strip_code_fences(s: str) -> str:
    s2 = (s or "").strip()
    if s2.startswith("```"):
        # Remove ```json ... ``` wrappers if they appear
        lines = s2.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()
    return s2


def call_gemini_for_object(
    client: "genai.Client",
    oid: Any,
    obj: Dict[str, Any],
    bounds: Dict[str, Any],
    reference_image: Optional["Image.Image"] = None,
) -> Dict[str, Any]:
    """
    Ask Gemini to estimate realistic physics parameters for the object, then
    post-validate so results are usable for simulation.
    """
    base_cfg = estimate_default_physics(obj, bounds)

    if client is None:
        return base_cfg

    prompt = make_gemini_prompt(oid, obj, bounds, base_cfg, has_image=reference_image is not None)

    try:
        # Use latest Gemini model for physics estimation
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-06-05")

        cfg_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
        }

        # Enable grounding if requested (disabled by default for physics estimation)
        grounding_enabled = os.getenv("GEMINI_GROUNDING_ENABLED", "false").lower() in {"1", "true", "yes"}
        if grounding_enabled:
            if hasattr(types, "Tool") and hasattr(types, "GoogleSearch"):
                try:
                    cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                except Exception:
                    pass  # Grounding not supported

        try:
            config = types.GenerateContentConfig(**cfg_kwargs)
        except Exception:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )

        contents = [prompt]
        if reference_image is not None:
            contents = [reference_image, prompt]

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        raw = _strip_code_fences(response.text or "")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")

        merged = dict(base_cfg)

        def _get_float(keys: List[str], default: float) -> float:
            for k in keys:
                if k in data:
                    try:
                        return float(data[k])
                    except Exception:
                        pass
            return float(default)

        def _get_bool(keys: List[str], default: bool) -> bool:
            for k in keys:
                if k in data and isinstance(data[k], bool):
                    return bool(data[k])
            return bool(default)

        def _get_str(keys: List[str], default: str) -> str:
            for k in keys:
                if k in data and isinstance(data[k], str) and data[k].strip():
                    return str(data[k]).strip()
            return str(default)

        # dynamic + collision shape
        merged["dynamic"] = _get_bool(["dynamic"], bool(merged.get("dynamic", True)))

        collision_shape = _get_str(["collision_shape", "collisionShape"], str(merged.get("collision_shape", "box"))).lower()
        if collision_shape not in {"box", "sphere", "capsule"}:
            # Keep it safe/stable: fall back to box collider
            collision_shape = "box"
        merged["collision_shape"] = collision_shape

        # restitution
        merged["restitution"] = _clamp(
            _get_float(["restitution", "bounce"], float(merged.get("restitution", 0.1))),
            0.0,
            1.0,
        )

        # friction: prefer explicit static/dynamic, else map from a single friction field
        static_f = _get_float(["static_friction", "staticFriction"], float(merged.get("static_friction", 0.6)))
        dynamic_f = _get_float(["dynamic_friction", "dynamicFriction"], float(merged.get("dynamic_friction", 0.5)))

        if "friction" in data and ("static_friction" not in data and "staticFriction" not in data and
                                  "dynamic_friction" not in data and "dynamicFriction" not in data):
            try:
                f = float(data["friction"])
                static_f = f * 1.1
                dynamic_f = f * 0.9
            except Exception:
                pass

        static_f = _clamp(static_f, 0.0, 2.0)
        dynamic_f = _clamp(dynamic_f, 0.0, 2.0)
        if dynamic_f > static_f:
            # enforce physically typical relationship
            dynamic_f = max(0.0, static_f - 0.05)

        merged["static_friction"] = static_f
        merged["dynamic_friction"] = dynamic_f

        # material name
        merged["material_name"] = _get_str(["material_name", "material"], str(merged.get("material_name", "generic")))

        # mass + bulk density, cross-checked with bounds volume when available
        volume = float(bounds.get("volume_m3") or 0.0)

        mass_val = _get_float(["mass_kg", "massKg", "mass"], float(merged.get("mass_kg", 1.0)))
        bulk_rho = _get_float(["bulk_density_kg_per_m3", "bulkDensity_kg_per_m3", "density_kg_per_m3", "density"], float(merged.get("bulk_density_kg_per_m3", 600.0)))

        # Keep density sane but wide (effective density can be low for hollow objects)
        bulk_rho = _clamp(bulk_rho, 0.5, 20000.0)

        # Either value could be missing/garbled; choose a stable mass:
        have_mass = "mass_kg" in data or "massKg" in data or "mass" in data
        have_rho = any(k in data for k in ["bulk_density_kg_per_m3", "bulkDensity_kg_per_m3", "density_kg_per_m3", "density"])

        mass_from_density = None
        if volume > 0.0 and have_rho:
            mass_from_density = bulk_rho * volume

        chosen_mass = mass_val
        if volume > 0.0 and mass_from_density is not None and have_mass:
            # If they disagree wildly, prefer density-derived mass (it tends to scale with size),
            # unless density-derived looks obviously broken.
            ratio = (mass_val / mass_from_density) if mass_from_density > 1e-9 else 1.0
            if ratio < 0.33 or ratio > 3.0:
                chosen_mass = mass_from_density
        elif volume > 0.0 and mass_from_density is not None and not have_mass:
            chosen_mass = mass_from_density

        # Clamp mass to wide safety bounds
        chosen_mass = _clamp(chosen_mass, 0.001, 1000.0)
        merged["mass_kg"] = float(chosen_mass)

        # Recompute effective bulk density for logging/consistency
        if volume > 0.0:
            merged["bulk_density_kg_per_m3"] = float(_clamp(chosen_mass / volume, 0.5, 20000.0))
        else:
            merged["bulk_density_kg_per_m3"] = float(bulk_rho)

        # Notes
        notes = _get_str(["notes"], "")
        extra = f"gemini_estimated; volume={volume:.6f} m^3; eff_bulk_density={merged['bulk_density_kg_per_m3']:.1f} kg/m^3"
        merged["notes"] = (notes + " | " + extra).strip(" |")

        # ========== ROBOTICS-FOCUSED PROPERTIES ==========

        # Get size for validation bounds
        size_m = list(bounds.get("size_m") or [0.1, 0.1, 0.1])

        # Semantic class
        semantic_class = _get_str(
            ["semantic_class", "semanticClass", "category", "object_type"],
            str(merged.get("semantic_class", "object"))
        ).lower().replace(" ", "_")
        # Validate against known classes
        valid_classes = {
            "object", "container", "tool", "food", "electronics", "furniture",
            "toy", "clothing", "book", "kitchenware", "bottle", "can", "box",
            "bag", "utensil", "appliance", "decoration", "plant", "sporting_goods",
            "office_supply", "hygiene", "cleaning", "storage", "hardware"
        }
        if semantic_class not in valid_classes:
            semantic_class = "object"
        merged["semantic_class"] = semantic_class

        # Center of mass offset
        def _get_float_list(keys: List[str], default: List[float], n: int = 3) -> List[float]:
            for k in keys:
                if k in data:
                    val = data[k]
                    if isinstance(val, (list, tuple)) and len(val) == n:
                        try:
                            return [float(x) for x in val]
                        except Exception:
                            pass
            return list(default)

        com_offset = _get_float_list(
            ["center_of_mass_offset", "centerOfMassOffset", "com_offset"],
            merged.get("center_of_mass_offset", [0.0, 0.0, 0.0])
        )
        # Clamp CoM offset to reasonable range (within half the object size)
        max_offset = max(size_m) * 0.5 if size_m else 0.1
        com_offset = [_clamp(v, -max_offset, max_offset) for v in com_offset]
        merged["center_of_mass_offset"] = com_offset

        # Graspable flag
        merged["graspable"] = _get_bool(
            ["graspable", "is_graspable", "pickable"],
            bool(merged.get("graspable", True))
        )
        # Auto-set graspable=False for very heavy objects
        if merged["mass_kg"] > 25.0:
            merged["graspable"] = False

        # Grasp regions
        grasp_regions_raw = data.get("grasp_regions") or data.get("graspRegions") or []
        grasp_regions: List[Dict[str, Any]] = []
        if isinstance(grasp_regions_raw, list):
            for gr in grasp_regions_raw[:5]:  # limit to 5 grasp regions
                if isinstance(gr, dict):
                    pos = gr.get("position") or gr.get("pos") or [0.0, 0.0, 0.0]
                    gtype = str(gr.get("type", "power")).lower()
                    width = float(gr.get("width_m", 0.08))
                    if gtype not in {"pinch", "power", "wrap", "suction"}:
                        gtype = "power"
                    if isinstance(pos, (list, tuple)) and len(pos) == 3:
                        try:
                            grasp_regions.append({
                                "position": [float(p) for p in pos],
                                "type": gtype,
                                "width_m": _clamp(width, 0.005, 0.30),
                            })
                        except Exception:
                            pass
        merged["grasp_regions"] = grasp_regions if grasp_regions else merged.get("grasp_regions", [])

        # Surface roughness
        merged["surface_roughness"] = _clamp(
            _get_float(["surface_roughness", "surfaceRoughness", "roughness"],
                      float(merged.get("surface_roughness", 0.5))),
            0.0,
            1.0,
        )

        # Contact offsets (PhysX-specific, use sensible defaults based on object size)
        min_dim = min(size_m) if size_m else 0.1
        default_contact = _clamp(min_dim * 0.02, 0.002, 0.01)  # 2% of smallest dimension
        default_rest = _clamp(min_dim * 0.005, 0.0005, 0.003)  # 0.5% of smallest dimension

        merged["contact_offset_m"] = _clamp(
            _get_float(["contact_offset_m", "contactOffset"], default_contact),
            0.001,
            0.02,
        )
        merged["rest_offset_m"] = _clamp(
            _get_float(["rest_offset_m", "restOffset"], default_rest),
            0.0001,
            0.005,
        )

        return merged

    except Exception as e:  # pragma: no cover
        print(f"[SIMREADY] WARNING: Gemini failed for obj {oid}: {e}", file=sys.stderr)
        return base_cfg


# ---------- Physics configuration builders ----------


def build_physics_config(
    obj: Dict[str, Any],
    bounds: Dict[str, Any],
    mesh_bounds: Optional[List[float]] = None,
    mesh_center: Optional[List[float]] = None,
    gemini_client: Optional["genai.Client"] = None,
    reference_image: Optional["Image.Image"] = None,
) -> Dict[str, Any]:
    """
    Build complete physics configuration for an object.

    This function orchestrates physics estimation using:
    1. Gemini AI (if client and reference image available)
    2. Fallback to heuristic defaults based on size/metadata

    Args:
        obj: Object dict with metadata
        bounds: Computed bounds dict with size_m, center_m, volume_m3
        mesh_bounds: Optional mesh bounds from metadata [w, h, d]
        mesh_center: Optional mesh center from metadata [x, y, z]
        gemini_client: Optional Gemini API client for AI estimation
        reference_image: Optional PIL image for Gemini

    Returns:
        Complete physics configuration dict
    """
    # Try Gemini-based estimation first
    if gemini_client is not None and have_gemini():
        try:
            physics_cfg = call_gemini_for_object(
                client=gemini_client,
                oid=obj.get("id"),
                obj=obj,
                bounds=bounds,
                reference_image=reference_image,
            )
            # Store mesh bounds info for catalog
            if mesh_bounds:
                physics_cfg["mesh_bounds"] = {
                    "size": mesh_bounds,
                    "center": mesh_center or [0.0, 0.0, 0.0],
                }
            return physics_cfg
        except Exception as e:
            print(f"[SIMREADY] WARNING: Gemini physics estimation failed: {e}", file=sys.stderr)

    # Fallback to heuristic defaults
    physics_cfg = estimate_default_physics(obj, bounds)

    # Store mesh bounds info for catalog
    if mesh_bounds:
        physics_cfg["mesh_bounds"] = {
            "size": mesh_bounds,
            "center": mesh_center or [0.0, 0.0, 0.0],
        }

    return physics_cfg


def emit_usd(
    visual_path: Path,
    physics_cfg: Dict[str, Any],
    bounds: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Emit a simready USD file for an asset.

    This function writes a USD wrapper with physics properties
    next to the visual asset.

    Args:
        visual_path: Path to the visual USD/USDZ asset
        physics_cfg: Physics configuration dict
        bounds: Optional bounds dict (extracted from physics_cfg if not provided)

    Returns:
        Path to the written simready.usda file
    """
    # Determine output path (sibling to visual asset)
    out_path = visual_path.parent / "simready.usda"

    # Compute asset reference relative to output location
    asset_rel = "./" + visual_path.name

    # Extract or compute bounds
    if bounds is None:
        # Try to reconstruct bounds from physics config
        mesh_bounds_info = physics_cfg.get("mesh_bounds", {})
        size = mesh_bounds_info.get("size") or [0.1, 0.1, 0.1]
        center = mesh_bounds_info.get("center") or [0.0, 0.0, 0.0]
        volume = size[0] * size[1] * size[2]
        bounds = {
            "size_m": size,
            "center_m": center,
            "volume_m3": volume,
            "source": "physics_cfg",
        }

    # Write the simready USD
    write_simready_usd(out_path, asset_rel, physics_cfg, bounds)

    return out_path


# ---------- USD writing ----------

def choose_static_visual_asset(assets_root: Path, oid: Any) -> Optional[Tuple[Path, str]]:
    """
    For a static object, pick the visual asset file to reference.

    Preference (USD-compatible formats only - GLB cannot be referenced directly):
    1) model.usdz
    2) asset.usdz
    3) model.usd / model.usdc
    """
    base_dir = assets_root / f"obj_{oid}"

    candidates = [
        base_dir / "model.usdz",
        base_dir / "asset.usdz",
        base_dir / "model.usd",
        base_dir / "model.usdc",
    ]

    for p in candidates:
        if p.is_file():
            rel = os.path.relpath(p, base_dir).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            return p, rel

    legacy_dir = assets_root / "static" / f"obj_{oid}"
    candidates = [
        legacy_dir / "model.usdz",
        legacy_dir / "asset.usdz",
        legacy_dir / "model.usd",
        legacy_dir / "model.usdc",
    ]
    for p in candidates:
        if p.is_file():
            rel = os.path.relpath(p, legacy_dir).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            return p, rel

    return None


def write_simready_usd(out_path: Path, asset_rel: str, physics: Dict[str, Any], bounds: Dict[str, Any]) -> None:
    """
    Create a small USD wrapper that is actually "sim-ready" in Isaac Sim:

    - If dynamic:
        - Apply PhysicsRigidBodyAPI + PhysicsMassAPI to /Asset
        - Author bool physics:rigidBodyEnabled
        - Author physics:mass (and do NOT also author density to avoid precedence confusion)
        - Author physics:centerOfMass for accurate manipulation dynamics

    - Always:
        - Reference the visual USD/Z under /Asset/Visual
        - Create a collision proxy prim with PhysicsCollisionAPI + PhysxCollisionAPI under /Asset/Collision
        - Define a UsdShade Material with PhysicsMaterialAPI under /Asset/Looks
        - Bind it to the collider using rel material:binding:physics
        - Add robotics-focused metadata (semantic class, graspability, grasp regions, etc.)
    """
    add_proxy = os.getenv("SIMREADY_ADD_PROXY_COLLIDER", "true").lower() in {"1", "true", "yes"}

    # Core physics properties
    dynamic = bool(physics.get("dynamic", True))
    mass = float(physics.get("mass_kg", 1.0))
    static_friction = float(physics.get("static_friction", 0.6))
    dynamic_friction = float(physics.get("dynamic_friction", 0.5))
    restitution = float(physics.get("restitution", 0.1))
    collision_shape = str(physics.get("collision_shape", "box")).lower()

    # Robotics-focused properties
    semantic_class = str(physics.get("semantic_class", "object"))
    material_name = str(physics.get("material_name", "generic"))
    graspable = bool(physics.get("graspable", True))
    grasp_regions = list(physics.get("grasp_regions", []))
    surface_roughness = float(physics.get("surface_roughness", 0.5))
    contact_offset = float(physics.get("contact_offset_m", 0.005))
    rest_offset = float(physics.get("rest_offset_m", 0.001))

    # Center of mass (geometric center + offset)
    com_offset = list(physics.get("center_of_mass_offset", [0.0, 0.0, 0.0]))

    size_m = list(bounds.get("size_m") or [0.1, 0.1, 0.1])
    center_m = list(bounds.get("center_m") or [0.0, 0.0, 0.0])

    # Compute absolute center of mass (geometric center + offset)
    center_of_mass = [center_m[i] + com_offset[i] for i in range(3)]

    # Use padded bounds for collision proxy (if any)
    size_pad = padded_size(size_m)

    lines: List[str] = []
    lines.append("#usda 1.0")
    lines.append("(\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n)")
    lines.append("")

    # Asset root
    if dynamic:
        lines.append('def Xform "Asset" (')
        lines.append('    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]')
        lines.append(")")
    else:
        # Static colliders don't need a rigid body; they are static when no RigidBodyAPI is in the ancestry.
        lines.append('def Xform "Asset"')
    lines.append("{")

    if dynamic:
        lines.append("    bool physics:rigidBodyEnabled = 1")
        lines.append(f"    float physics:mass = {mass:.6f}")
        # Center of mass for accurate manipulation dynamics
        lines.append(f"    point3f physics:centerOfMass = ({center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f})")
        # (Intentionally not authoring physics:density here. Mass has precedence over density in USD Physics.)

    # Visual reference
    lines.append("")
    lines.append('    def Xform "Visual" (')
    lines.append(f"        prepend references = @{asset_rel}@")
    lines.append("    )")
    lines.append("    {")
    lines.append("    }")

    # Physics material (UsdShade.Material + PhysicsMaterialAPI)
    lines.append("")
    lines.append('    def Scope "Looks"')
    lines.append("    {")
    lines.append('        def Material "PhysicsMaterial" (')
    lines.append('            prepend apiSchemas = ["PhysicsMaterialAPI"]')
    lines.append("        )")
    lines.append("        {")
    lines.append(f"            float physics:staticFriction = {static_friction:.4f}")
    lines.append(f"            float physics:dynamicFriction = {dynamic_friction:.4f}")
    lines.append(f"            float physics:restitution = {restitution:.4f}")
    lines.append("        }")
    lines.append("    }")

    # Collision proxy
    if add_proxy:
        lines.append("")
        lines.append('    def Xform "Collision"')
        lines.append("    {")
        material_path = "</Asset/Looks/PhysicsMaterial>"

        if collision_shape == "sphere":
            radius = 0.5 * max(size_pad)
            lines.append('        def Sphere "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append(f"            double radius = {radius:.6f}")
            lines.append(f"            double3 xformOp:translate = ({center_m[0]:.6f}, {center_m[1]:.6f}, {center_m[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            # PhysX contact properties for stable simulation
            lines.append(f"            float physxCollision:contactOffset = {contact_offset:.6f}")
            lines.append(f"            float physxCollision:restOffset = {rest_offset:.6f}")
            lines.append("        }")

        elif collision_shape == "capsule":
            # Capsule attributes: radius + height (shaft, excludes caps) + axis
            # Choose axis along the largest dimension
            sx, sy, sz = size_pad
            dims = {"X": sx, "Y": sy, "Z": sz}
            axis = max(dims, key=dims.get)
            # Radius based on the smaller transverse dimensions
            if axis == "X":
                r = 0.5 * min(sy, sz)
                max_dim = sx
            elif axis == "Y":
                r = 0.5 * min(sx, sz)
                max_dim = sy
            else:
                r = 0.5 * min(sx, sy)
                max_dim = sz

            r = max(r, 1e-4)
            height = max(max_dim - 2.0 * r, 0.0)

            lines.append('        def Capsule "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append(f'            uniform token axis = "{axis}"')
            lines.append(f"            double radius = {r:.6f}")
            lines.append(f"            double height = {height:.6f}")
            lines.append(f"            double3 xformOp:translate = ({center_m[0]:.6f}, {center_m[1]:.6f}, {center_m[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            # PhysX contact properties for stable simulation
            lines.append(f"            float physxCollision:contactOffset = {contact_offset:.6f}")
            lines.append(f"            float physxCollision:restOffset = {rest_offset:.6f}")
            lines.append("        }")

        else:
            # Default: box proxy collider (Cube) with non-uniform scale.
            lines.append('        def Cube "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append("            double size = 1")
            lines.append(f"            double3 xformOp:translate = ({center_m[0]:.6f}, {center_m[1]:.6f}, {center_m[2]:.6f})")
            lines.append(f"            float3 xformOp:scale = ({size_pad[0]:.6f}, {size_pad[1]:.6f}, {size_pad[2]:.6f})")
            # xformOpOrder is applied in reverse; this ordering yields scale first, then translate.
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            # PhysX contact properties for stable simulation
            lines.append(f"            float physxCollision:contactOffset = {contact_offset:.6f}")
            lines.append(f"            float physxCollision:restOffset = {rest_offset:.6f}")
            lines.append("        }")

        lines.append("    }")

    # ========== ROBOTICS-FOCUSED METADATA ==========
    lines.append("")
    lines.append("    # Robotics metadata for perception and manipulation")

    # Semantic class (for object detection/segmentation)
    lines.append(f'    string semantic:class = "{semantic_class}"')

    # Material name (for tactile/visual sensing)
    safe_material = material_name.replace('"', '\\"')
    lines.append(f'    string simready:material = "{safe_material}"')

    # Graspability flag
    graspable_val = "true" if graspable else "false"
    lines.append(f"    bool simready:graspable = {graspable_val}")

    # Surface roughness (for tactile simulation: 0=smooth glass, 1=sandpaper)
    lines.append(f"    float simready:surfaceRoughness = {surface_roughness:.4f}")

    # Grasp regions (for manipulation planning)
    if grasp_regions:
        lines.append("")
        lines.append('    def Scope "GraspRegions"')
        lines.append("    {")
        for idx, gr in enumerate(grasp_regions):
            pos = gr.get("position", [0.0, 0.0, 0.0])
            gtype = gr.get("type", "power")
            width = gr.get("width_m", 0.08)
            lines.append(f'        def Xform "Grasp_{idx}" {{')
            lines.append(f"            double3 xformOp:translate = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append(f'            string simready:graspType = "{gtype}"')
            lines.append(f"            float simready:graspWidth = {width:.4f}")
            lines.append("        }")
        lines.append("    }")

    # Optional notes for debugging
    notes = str(physics.get("notes", "")).strip()
    if notes:
        safe_notes = notes.replace('"', '\\"')
        lines.append("")
        lines.append(f'    string simready:notes = "{safe_notes}"')

    lines.append("}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------- Main pipeline ----------


def prepare_simready_assets_job(
    bucket: str,
    scene_id: str,
    assets_prefix: str,
    root: Path = GCS_ROOT,
) -> int:
    if not assets_prefix:
        print("[SIMREADY] ASSETS_PREFIX is required", file=sys.stderr)
        return 1

    assets_root = root / assets_prefix
    manifest_path = assets_root / "scene_manifest.json"

    print(f"[SIMREADY] Bucket={bucket}")
    print(f"[SIMREADY] Scene={scene_id}")
    print(f"[SIMREADY] Assets root={assets_root}")
    print(f"[SIMREADY] Loading {manifest_path} (or legacy scene_assets.json)")

    scene_assets = load_manifest_or_scene_assets(assets_root)
    if scene_assets is None:
        legacy_path = assets_root / "scene_assets.json"
        print(
            f"[SIMREADY] scene manifest missing at {manifest_path} and {legacy_path}",
            file=sys.stderr,
        )
        return 1
    objects = scene_assets.get("objects", [])
    print(f"[SIMREADY] Found {len(objects)} objects in manifest")

    catalog_client = AssetCatalogClient()

    client = None
    if have_gemini():
        api_key = get_gemini_api_key()
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-06-05")
        print(f"[SIMREADY] Gemini client initialized (model: {model_name})")
    else:
        print(
            "[SIMREADY] WARNING: No Gemini API key found!",
            file=sys.stderr,
        )
        print(
            "[SIMREADY] Set GOOGLE_API_KEY or GEMINI_API_KEY for AI-powered physics estimation.",
            file=sys.stderr,
        )
        print(
            "[SIMREADY] Falling back to heuristics (degraded quality).",
            file=sys.stderr,
        )

    simready_paths: Dict[Any, str] = {}

    for obj in objects:
        oid = obj.get("id")
        if oid is None:
            continue

        print(f"[SIMREADY] Processing obj {oid}")

        visual = choose_static_visual_asset(assets_root, oid)
        if visual is None:
            print(f"[SIMREADY] WARNING: no visual asset found for obj {oid}", file=sys.stderr)
            continue

        visual_path, visual_rel = visual
        obj_metadata = load_object_metadata(root, obj, assets_prefix, catalog_client)
        obj_metadata = obj_metadata or {}

        mesh_bounds_metadata, mesh_center_metadata = extract_mesh_bounds_from_metadata(
            obj_metadata
        )
        obb_bounds, obb_center = extract_obb_bounds_from_obj(obj)

        gemini_estimated_size = None
        ref_img = None
        ref_image = choose_reference_image_path(root, obj)
        if ref_image:
            ref_img = load_image_for_gemini(ref_image)
            if client and ref_img:
                gemini_estimated_size = estimate_scale_gemini(
                    client, obj, ref_img, obj_metadata.get("class_name")
                )

        # Compute bounds using metadata and Gemini estimates
        bounds = compute_bounds(
            obj=obj,
            metadata=obj_metadata,
            gemini_estimated_size=gemini_estimated_size,
        )

        # Build physics config with Gemini AI or heuristics
        physics_cfg = build_physics_config(
            obj=obj,
            bounds=bounds,
            mesh_bounds=mesh_bounds_metadata,
            mesh_center=mesh_center_metadata,
            gemini_client=client,
            reference_image=ref_img,
        )

        # Compute sim2real distribution ranges for domain randomization
        physics_distributions = compute_physics_distributions(physics_cfg, bounds)
        physics_cfg.update(physics_distributions)

        # Emit the simready USD with physics
        sim_path = emit_usd(visual_path, physics_cfg, bounds)

        if catalog_client:
            try:
                export_bounds = physics_cfg.get("mesh_bounds") or {}
                catalog_payload = {
                    "id": obj.get("id"),
                    "class_name": obj.get("class_name"),
                    "asset_path": obj.get("asset_path") or visual_rel,
                    "mesh_bounds": {"export": export_bounds},
                    "physics": physics_cfg,
                    "material_name": physics_cfg.get("material_name"),
                }
                catalog_client.publish_metadata(
                    oid, catalog_payload, asset_path=catalog_payload["asset_path"]
                )
            except Exception as exc:  # pragma: no cover - network errors
                print(
                    f"[SIMREADY] WARNING: failed to publish catalog metadata for obj {oid}: {exc}",
                    file=sys.stderr,
                )

        sim_rel = f"{assets_prefix}/obj_{oid}/simready.usda"
        if "static/obj_" in str(visual_path):
            sim_rel = f"{assets_prefix}/static/obj_{oid}/simready.usda"

        print(f"[SIMREADY] Wrote simready asset for obj {oid} -> {sim_path}")
        simready_paths[oid] = sim_rel

    marker_path = assets_root / ".simready_complete"
    if simready_paths:
        marker_content = {
            "status": "complete",
            "simready_assets": simready_paths,
            "count": len(simready_paths),
        }
        marker_path.write_text(json.dumps(marker_content, indent=2), encoding="utf-8")
        print(f"[SIMREADY] Created completion marker at {marker_path}")
        print(f"[SIMREADY] Generated {len(simready_paths)} simready assets")
    else:
        marker_content = {
            "status": "complete",
            "simready_assets": {},
            "count": 0,
            "note": "No objects required simready processing (no visual assets found)",
        }
        marker_path.write_text(json.dumps(marker_content, indent=2), encoding="utf-8")
        print(
            "[SIMREADY] No simready assets were created; created completion marker anyway",
            file=sys.stderr,
        )

    return 0


def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

    from blueprint_sim.simready import run_from_env

    sys.exit(run_from_env(root=GCS_ROOT))


if __name__ == "__main__":
    main()
