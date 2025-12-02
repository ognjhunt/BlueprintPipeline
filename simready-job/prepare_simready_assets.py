import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    # Google GenAI SDK for Gemini 3.x
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

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


def load_object_metadata(root: Path, obj: Dict[str, Any], assets_prefix: str) -> Optional[dict]:
    """
    Same lookup strategy as usd-assembly build_scene_usd:
    1. explicit metadata_path
    2. next to asset_path
    3. fallback: assets/static/obj_{id}/metadata.json
    """
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


def extract_mesh_size_from_metadata(metadata: Optional[dict]) -> Optional[List[float]]:
    """
    Try to read an axis-aligned bounding box size [sx, sy, sz] from the
    mesh metadata, in *meters*. This is whatever the rest of the pipeline
    wrote under mesh_bounds.export.size (or similar).
    """
    if not metadata:
        return None
    mesh_bounds = metadata.get("mesh_bounds") or {}
    export_bounds = mesh_bounds.get("export") or mesh_bounds.get("bounds") or mesh_bounds
    size = export_bounds.get("size")
    if size and len(size) == 3:
        try:
            return [float(s) for s in size]
        except Exception:
            return None
    return None


# ---------- Category heuristics ----------

# Tokens -> canonical category label
CATEGORY_KEYWORDS: List[Tuple[str, str]] = [
    # furniture
    ("sectional", "sofa"),
    ("loveseat", "sofa"),
    ("sofa", "sofa"),
    ("couch", "sofa"),
    ("armchair", "chair"),
    ("dining chair", "chair"),
    ("office chair", "chair"),
    ("stool", "chair"),
    ("bench", "chair"),
    ("chair", "chair"),
    ("coffee table", "table"),
    ("end table", "table"),
    ("side table", "table"),
    ("nightstand", "table"),
    ("desk", "table"),
    ("table", "table"),
    ("bed", "bed"),
    ("headboard", "bed"),
    ("dresser", "cabinet"),
    ("cabinet", "cabinet"),
    ("shelf", "shelf"),
    ("bookcase", "shelf"),
    # soft goods / clothing
    ("throw pillow", "pillow"),
    ("pillow", "pillow"),
    ("cushion", "pillow"),
    ("blanket", "textile"),
    ("comforter", "textile"),
    ("duvet", "textile"),
    ("towel", "textile"),
    ("curtain", "textile"),
    ("sheet", "textile"),
    ("hat", "hat"),
    ("beanie", "hat"),
    ("cap", "hat"),
    ("shoe", "shoe"),
    ("sneaker", "shoe"),
    ("boot", "shoe"),
    # containers / kitchen
    ("mug", "mug"),
    ("coffee cup", "mug"),
    ("teacup", "mug"),
    ("cup", "mug"),
    ("bowl", "bowl"),
    ("plate", "plate"),
    ("dish", "plate"),
    ("bottle", "bottle"),
    ("jar", "bottle"),
    ("vase", "vase"),
    # cutlery
    ("spoon", "spoon"),
    ("teaspoon", "spoon"),
    ("tablespoon", "spoon"),
    ("fork", "fork"),
    ("knife", "knife"),
    ("butter knife", "knife"),
    ("spatula", "spatula"),
    # appliances
    ("microwave", "microwave"),
    ("toaster", "appliance"),
    ("blender", "appliance"),
    ("coffee maker", "appliance"),
    # electronics / devices
    ("laptop", "laptop"),
    ("notebook computer", "laptop"),
    ("monitor", "monitor"),
    ("tv", "monitor"),
    ("television", "monitor"),
    ("keyboard", "keyboard"),
    ("mouse", "mouse"),
    ("phone", "phone"),
    ("smartphone", "phone"),
    ("tablet", "tablet"),
    # misc small stuff
    ("book", "book"),
    ("box", "box"),
    ("basket", "basket"),
    ("backpack", "bag"),
    ("handbag", "bag"),
    ("bag", "bag"),
    ("plant", "plant"),
]

# Per-category "effective" priors in terms of the object's *bounding box* volume.
# These are NOT true material densities; they are tuned so that a typical real-world
# object with a reasonably scaled bounding box lands in the right ballpark for mass.
CATEGORY_PRIORS: Dict[str, Dict[str, Any]] = {
    "generic": {
        "density_kg_per_m3": 400.0,
        "mass_range_kg": (0.1, 10.0),
        "friction": 0.9,
        "restitution": 0.2,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "generic",
    },
    "sofa": {
        "density_kg_per_m3": 30.0,
        "mass_range_kg": (20.0, 80.0),
        "friction": 0.9,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "upholstery",
    },
    "chair": {
        "density_kg_per_m3": 50.0,
        "mass_range_kg": (5.0, 25.0),
        "friction": 0.9,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "wood_fabric",
    },
    "table": {
        "density_kg_per_m3": 70.0,
        "mass_range_kg": (10.0, 40.0),
        "friction": 0.8,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "wood",
    },
    "bed": {
        "density_kg_per_m3": 40.0,
        "mass_range_kg": (25.0, 100.0),
        "friction": 0.9,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "mattress",
    },
    "cabinet": {
        "density_kg_per_m3": 120.0,
        "mass_range_kg": (20.0, 80.0),
        "friction": 0.7,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "wood",
    },
    "shelf": {
        "density_kg_per_m3": 80.0,
        "mass_range_kg": (10.0, 60.0),
        "friction": 0.7,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "wood",
    },
    "pillow": {
        "density_kg_per_m3": 10.0,
        "mass_range_kg": (0.2, 2.0),
        "friction": 1.2,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "fabric",
    },
    "textile": {
        "density_kg_per_m3": 5.0,
        "mass_range_kg": (0.2, 3.0),
        "friction": 1.3,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "fabric",
    },
    "hat": {
        "density_kg_per_m3": 8.0,
        "mass_range_kg": (0.05, 0.3),
        "friction": 1.2,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "fabric",
    },
    "shoe": {
        "density_kg_per_m3": 80.0,
        "mass_range_kg": (0.2, 1.5),
        "friction": 1.1,
        "restitution": 0.15,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "rubber",
    },
    "mug": {
        "density_kg_per_m3": 300.0,
        "mass_range_kg": (0.25, 0.8),
        "friction": 0.6,
        "restitution": 0.2,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "ceramic",
    },
    "bowl": {
        "density_kg_per_m3": 250.0,
        "mass_range_kg": (0.2, 1.0),
        "friction": 0.6,
        "restitution": 0.2,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "ceramic",
    },
    "plate": {
        "density_kg_per_m3": 250.0,
        "mass_range_kg": (0.2, 1.0),
        "friction": 0.6,
        "restitution": 0.15,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "ceramic",
    },
    "bottle": {
        "density_kg_per_m3": 80.0,
        "mass_range_kg": (0.1, 1.5),
        "friction": 0.5,
        "restitution": 0.25,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "plastic_glass",
    },
    "vase": {
        "density_kg_per_m3": 150.0,
        "mass_range_kg": (0.5, 5.0),
        "friction": 0.5,
        "restitution": 0.25,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "ceramic_glass",
    },
    "laptop": {
        "density_kg_per_m3": 200.0,
        "mass_range_kg": (1.0, 3.0),
        "friction": 0.5,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "metal_plastic",
    },
    "monitor": {
        "density_kg_per_m3": 150.0,
        "mass_range_kg": (2.0, 10.0),
        "friction": 0.5,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "glass_plastic",
    },
    "keyboard": {
        "density_kg_per_m3": 150.0,
        "mass_range_kg": (0.3, 1.5),
        "friction": 0.7,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "plastic",
    },
    "mouse": {
        "density_kg_per_m3": 200.0,
        "mass_range_kg": (0.05, 0.3),
        "friction": 0.5,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "plastic",
    },
    "phone": {
        "density_kg_per_m3": 300.0,
        "mass_range_kg": (0.1, 0.4),
        "friction": 0.7,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "glass_plastic",
    },
    "tablet": {
        "density_kg_per_m3": 250.0,
        "mass_range_kg": (0.3, 1.0),
        "friction": 0.7,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "glass_plastic",
    },
    "book": {
        "density_kg_per_m3": 300.0,
        "mass_range_kg": (0.2, 2.0),
        "friction": 0.9,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "paper",
    },
    "box": {
        "density_kg_per_m3": 60.0,
        "mass_range_kg": (0.1, 10.0),
        "friction": 0.8,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "cardboard",
    },
    "basket": {
        "density_kg_per_m3": 40.0,
        "mass_range_kg": (0.1, 5.0),
        "friction": 0.9,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "wicker",
    },
    "bag": {
        "density_kg_per_m3": 30.0,
        "mass_range_kg": (0.1, 3.0),
        "friction": 1.0,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "fabric",
    },
    "plant": {
        "density_kg_per_m3": 60.0,
        "mass_range_kg": (0.2, 10.0),
        "friction": 0.8,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "plant",
    },
    "spoon": {
        "density_kg_per_m3": 600.0,
        "mass_range_kg": (0.02, 0.08),
        "friction": 0.4,
        "restitution": 0.15,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "stainless_steel",
    },
    "fork": {
        "density_kg_per_m3": 650.0,
        "mass_range_kg": (0.025, 0.09),
        "friction": 0.4,
        "restitution": 0.15,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "stainless_steel",
    },
    "knife": {
        "density_kg_per_m3": 700.0,
        "mass_range_kg": (0.03, 0.12),
        "friction": 0.35,
        "restitution": 0.15,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "stainless_steel",
    },
    "spatula": {
        "density_kg_per_m3": 500.0,
        "mass_range_kg": (0.05, 0.15),
        "friction": 0.5,
        "restitution": 0.1,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "metal_plastic",
    },
    "microwave": {
        "density_kg_per_m3": 100.0,
        "mass_range_kg": (10.0, 25.0),
        "friction": 0.6,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "metal_plastic",
    },
    "appliance": {
        "density_kg_per_m3": 120.0,
        "mass_range_kg": (1.0, 15.0),
        "friction": 0.6,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "metal_plastic",
    },
    "large_furniture": {
        "density_kg_per_m3": 40.0,
        "mass_range_kg": (20.0, 150.0),
        "friction": 0.8,
        "restitution": 0.05,
        "dynamic": True,
        "collision_shape": "box",
        "material_name": "wood",
    },
    "small_object": {
        "density_kg_per_m3": 200.0,
        "mass_range_kg": (0.05, 1.0),
        "friction": 0.7,
        "restitution": 0.2,
        "dynamic": True,
        "collision_shape": "mesh",
        "material_name": "generic_small",
    },
}


def classify_category(obj: Dict[str, Any], metadata: Optional[dict]) -> str:
    """
    Try to map the object to a semantic category like 'sofa', 'hat', 'mug', etc.,
    based on class_name / label text and (optionally) approximate size.
    """
    text_bits: List[str] = []
    for key in ("class_name", "category", "label", "name"):
        val = obj.get(key)
        if isinstance(val, str):
            text_bits.append(val.lower())

    if metadata:
        for key in ("class_name", "category", "label", "name"):
            val = metadata.get(key)
            if isinstance(val, str):
                text_bits.append(val.lower())

    blob = " ".join(text_bits)

    for token, cat in CATEGORY_KEYWORDS:
        if token in blob:
            return cat

    # Fall back to a size-based guess if we have bounds.
    size = extract_mesh_size_from_metadata(metadata)
    if size:
        max_dim = max(size)
        if max_dim > 1.5:
            return "large_furniture"
        if max_dim < 0.3:
            return "small_object"

    return "generic"


def get_category_prior(category: str) -> Dict[str, Any]:
    return CATEGORY_PRIORS.get(category, CATEGORY_PRIORS["generic"])


# ---------- Physics config (default + Gemini) ----------

def estimate_default_physics(obj: Dict[str, Any], metadata: Optional[dict]) -> Dict[str, Any]:
    """
    Heuristic base estimate for physics that is then refined by Gemini.
    We combine:
      - the object's approximate bounding box volume
      - a category-specific effective density and mass range
    """
    category = classify_category(obj, metadata)
    prior = get_category_prior(category)

    size = extract_mesh_size_from_metadata(metadata)
    volume = None
    if size:
        try:
            sx, sy, sz = [max(float(s), 1e-3) for s in size]
            volume = sx * sy * sz  # m^3
        except Exception:
            volume = None

    rho = float(prior.get("density_kg_per_m3", 400.0))
    mass_min, mass_max = prior.get("mass_range_kg", (0.1, 10.0))
    mass: float
    if volume is not None:
        mass_est = volume * rho
        # Clip to reasonable range for this category
        mass = float(np.clip(mass_est, mass_min, mass_max))
    else:
        # If we know nothing about volume, use the middle of the prior range.
        mass = float(0.5 * (mass_min + mass_max))

    dynamic = bool(prior.get("dynamic", True))
    friction = float(prior.get("friction", 0.9))
    restitution = float(prior.get("restitution", 0.2))
    collision_shape = str(prior.get("collision_shape", "mesh"))
    material_name = str(prior.get("material_name", "generic"))

    note_parts = [f"category={category}", f"rho={rho:.2f} kg/m^3"]
    if volume is not None:
        note_parts.append(f"volume={volume:.4f} m^3")
    note_parts.append(f"mass_range=[{mass_min:.2f},{mass_max:.2f}]")
    notes = "; ".join(note_parts)

    return {
        "dynamic": dynamic,
        "collisionShape": collision_shape,
        "restitution": restitution,
        "friction": friction,
        "mass_kg": mass,
        "density_kg_per_m3": rho,
        "material_name": material_name,
        "notes": notes,
    }


def make_gemini_prompt(
    oid: Any,
    obj: Dict[str, Any],
    metadata: Optional[dict],
    base_cfg: Dict[str, Any],
) -> str:
    """
    Prompt Gemini with:
      - semantic info (class_name, type, pipeline)
      - approximate geometric info (obb extents, mesh size / volume)
      - our heuristic base physics estimate
    and ask it to refine the physics parameters.
    """
    size = extract_mesh_size_from_metadata(metadata)
    volume = None
    if size:
        try:
            sx, sy, sz = [max(float(s), 1e-3) for s in size]
            volume = sx * sy * sz
        except Exception:
            volume = None

    minimal = {
        "id": oid,
        "type": obj.get("type"),
        "class_name": obj.get("class_name"),
        "pipeline": obj.get("pipeline"),
        "obb_extents": (obj.get("obb") or {}).get("extents"),
        "mesh_size_m": size,
        "approx_volume_m3": volume,
        "base_physics_estimate": base_cfg,
    }

    skeleton = {
        "dynamic": base_cfg.get("dynamic", True),
        "collisionShape": base_cfg.get("collisionShape", "mesh"),
        "restitution": base_cfg.get("restitution", 0.2),
        "friction": base_cfg.get("friction", 0.9),
        "mass_kg": base_cfg.get("mass_kg", 1.0),
        "density_kg_per_m3": base_cfg.get("density_kg_per_m3", 400.0),
        "material_name": base_cfg.get("material_name", "generic"),
        "notes": "",
    }

    prompt = f"""
You are helping configure 3D assets for robotics training in NVIDIA Isaac Sim / USD.

Given object metadata and a heuristic physics estimate, refine the physics so
the object behaves as realistically as possible in the real world.

IMPORTANT: Use your knowledge and grounding capabilities to provide accurate, real-world
physics parameters for each specific object type. Every object should have realistic,
distinct properties appropriate to its category and material composition:
- A spoon and fork should have similar but subtly different properties (mass, material)
- A microwave should be significantly heavier with metal/plastic properties
- A blanket should have low mass, high friction, fabric properties
- Consider the actual real-world physics of each object type

The simulation uses:
- meters for linear distance
- kilograms for mass
- rigid bodies only (no joints in this step)

Key goals:
- Mass should be in a realistic range for the object's category and size based on
  real-world examples (small hat < 0.3 kg, typical sofa ~20–80 kg, mug ~0.25–0.8 kg,
  spoon ~0.03 kg, fork ~0.04 kg, microwave ~15-20 kg, blanket ~1-2 kg, etc.).
- Friction should be consistent with the actual material composition:
  fabric/rubber > 1, wood ~0.8, ceramic/glass/plastic ~0.4–0.8, metal ~0.3–0.6.
- Restitution (bounciness) is usually low (0–0.3) for household objects.
- collisionShape should be "box" for boxy furniture, "mesh" for irregular shapes,
  or "sphere"/"capsule" when that is a good fit.
- dynamic is true for things that should move (almost everything except walls/floors).

Return ONLY valid JSON (no comments, no extra text) that matches this structure:

{skeleton!r}

Fields:
- dynamic: true if the object should move under physics.
- collisionShape: "box", "sphere", "capsule", or "mesh".
- restitution: 0 (no bounce) to 1 (very bouncy).
- friction: 0 (very slippery) to 2 (very sticky).
- mass_kg: positive float in kilograms (use real-world values for this specific object type).
- density_kg_per_m3: effective bulk density (mass / bounding-box volume).
- material_name: short label like "wood", "metal", "plastic", "rubber", "fabric", "ceramic", "stainless_steel".
- notes: very short explanation of your choices and any real-world considerations.

Here is the metadata and base estimate for this object:

{json.dumps(minimal, indent=2)}
"""
    return prompt


def have_gemini() -> bool:
    return genai is not None and types is not None and bool(os.getenv("GEMINI_API_KEY"))


def call_gemini_for_object(
    client: "genai.Client",
    oid: Any,
    obj: Dict[str, Any],
    metadata: Optional[dict],
) -> Dict[str, Any]:
    """
    Ask Gemini to refine a heuristic physics estimate. We always start from
    category-based priors, then let Gemini adjust within reasonable bounds.
    """
    base_cfg = estimate_default_physics(obj, metadata)

    if client is None:
        return base_cfg

    # Determine category + prior so we can clamp Gemini's suggestions.
    category = classify_category(obj, metadata)
    prior = get_category_prior(category)
    mass_min, mass_max = prior.get("mass_range_kg", (0.1, 10.0))

    prompt = make_gemini_prompt(oid, obj, metadata, base_cfg)

    try:
        # Decide which model we're calling - default to Gemini 3.0 Pro for realistic simready
        model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

        # Build a GenerateContentConfig that works for both 2.5 and 3.x
        cfg_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
        }

        # Enable grounding for Gemini 3.x models (default: enabled)
        grounding_enabled = os.getenv("GEMINI_GROUNDING_ENABLED", "true").lower() in {"1", "true", "yes"}
        if model_name.startswith("gemini-3") and grounding_enabled:
            if hasattr(types, "GroundingConfig") and hasattr(types, "GoogleSearch"):
                cfg_kwargs["grounding"] = types.GroundingConfig(
                    google_search=types.GoogleSearch()
                )

        # Only use thinking_config when the SDK exposes it
        if hasattr(types, "ThinkingConfig"):
            ThinkingConfig = getattr(types, "ThinkingConfig")
            ThinkingLevel = getattr(types, "ThinkingLevel", None)

            if model_name.startswith("gemini-3") and ThinkingLevel is not None:
                # Gemini 3: use thinking_level
                cfg_kwargs["thinking_config"] = ThinkingConfig(
                    thinking_level=getattr(ThinkingLevel, "HIGH", "HIGH")
                )
            elif model_name.startswith("gemini-2.5"):
                # Gemini 2.5: use include_thoughts-style config if available
                cfg_kwargs["thinking_config"] = ThinkingConfig(
                    include_thoughts=True
                )

        try:
            config = types.GenerateContentConfig(**cfg_kwargs)
        except Exception:
            # If anything about thinking_config blows up, fall back to
            # plain JSON mode with no thinking.
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )

        # Make the request with the resolved model_name
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        raw = response.text or ""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")

        # Merge onto defaults; ignore unknown keys, and clamp to safe ranges.
        merged = dict(base_cfg)

        # Helper to clamp floats safely
        def _get_clamped(key: str, default: float, lo: float, hi: float) -> float:
            try:
                val = float(data.get(key, default))
            except Exception:
                return default
            return float(min(max(val, lo), hi))

        if "dynamic" in data and isinstance(data["dynamic"], bool):
            merged["dynamic"] = data["dynamic"]

        if "collisionShape" in data and isinstance(data["collisionShape"], str):
            merged["collisionShape"] = data["collisionShape"]

        # Friction/restitution
        merged["friction"] = _get_clamped(
            "friction", merged.get("friction", 0.9), 0.0, 2.0
        )
        merged["restitution"] = _get_clamped(
            "restitution", merged.get("restitution", 0.2), 0.0, 1.0
        )

        # Mass + density
        mass_default = float(merged.get("mass_kg", 1.0))
        merged["mass_kg"] = _get_clamped(
            "mass_kg", mass_default, float(mass_min), float(mass_max)
        )

        # density_kg_per_m3 should be positive but not absurd
        try:
            rho_val = float(data.get("density_kg_per_m3", merged.get("density_kg_per_m3", 400.0)))
            if rho_val > 0.0 and rho_val < 5000.0:
                merged["density_kg_per_m3"] = rho_val
        except Exception:
            pass

        if "material_name" in data and isinstance(data["material_name"], str):
            merged["material_name"] = data["material_name"]

        if "notes" in data and isinstance(data["notes"], str):
            merged["notes"] = data["notes"]

        # Append a short note that we clamped mass to priors.
        merged["notes"] = (
            merged.get("notes", "")
            + f" | gemini_refined category={category}, mass_range=[{mass_min},{mass_max}]"
        ).strip()

        return merged

    except Exception as e:  # pragma: no cover
        print(f"[SIMREADY] WARNING: Gemini failed for obj {oid}: {e}", file=sys.stderr)
        return base_cfg


# ---------- USD writing ----------

def choose_static_visual_asset(assets_root: Path, oid: Any) -> Optional[Tuple[Path, str]]:
    """
    For a static object, pick the visual asset file to reference.

    Preference:
    1) model.usdz
    2) model.usd / model.usdc
    3) asset.glb
    4) mesh.glb
    """
    # Newer pipeline: assets/obj_{id}/...
    base_dir = assets_root / f"obj_{oid}"

    candidates = [
        base_dir / "model.usdz",
        base_dir / "model.usd",
        base_dir / "model.usdc",
        base_dir / "asset.glb",
        base_dir / "mesh.glb",
    ]

    for p in candidates:
        if p.is_file():
            rel = os.path.relpath(p, base_dir).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            return p, rel

    # Older layout: assets/static/obj_{id}/...
    legacy_dir = assets_root / "static" / f"obj_{oid}"
    candidates = [
        legacy_dir / "model.usdz",
        legacy_dir / "model.usd",
        legacy_dir / "model.usdc",
        legacy_dir / "asset.glb",
        legacy_dir / "mesh.glb",
    ]
    for p in candidates:
        if p.is_file():
            rel = os.path.relpath(p, legacy_dir).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            return p, rel

    return None


def write_simready_usd(out_path: Path, asset_rel: str, physics: Dict[str, Any]) -> None:
    """
    Create a small USD wrapper that:
    - makes the object a rigid body
    - adds mass and (optionally) density
    - encodes friction/restitution via PhysicsMaterialAPI
    - references the visual asset stage.

    We keep everything on a single "Asset" prim so that it is easy to swap into
    downstream USD scenes.
    """
    mass = float(physics.get("mass_kg", 1.0))
    dynamic = bool(physics.get("dynamic", True))
    density = float(physics.get("density_kg_per_m3", 400.0))
    friction = float(physics.get("friction", 0.9))
    restitution = float(physics.get("restitution", 0.2))

    # Map single "friction" into static/dynamic; static a bit higher.
    static_friction = max(min(friction * 1.1, 2.0), 0.0)
    dynamic_friction = max(min(friction * 0.9, 2.0), 0.0)

    enabled_token = "true" if dynamic else "false"

    lines: List[str] = []
    lines.append("#usda 1.0")
    lines.append("(\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n)")
    lines.append("")
    lines.append('def Xform "Asset" (')
    lines.append(
        '    prepend apiSchemas = ['
        '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI", "PhysicsMaterialAPI"'
        "]"
    )
    lines.append(")")
    lines.append("{")
    lines.append(f"    float physics:mass = {mass:.6f}")
    lines.append(f"    float physics:density = {density:.6f}")
    lines.append(f"    float physics:staticFriction = {static_friction:.4f}")
    lines.append(f"    float physics:dynamicFriction = {dynamic_friction:.4f}")
    lines.append(f"    float physics:restitution = {restitution:.4f}")
    lines.append(f'    token physics:rigidBodyEnabled = "{enabled_token}"')
    lines.append("")
    lines.append('    def Xform "Visual" {')
    lines.append(f"        rel references = @{asset_rel}@")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------- Main pipeline ----------

def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX")  # scenes/<sceneId>/assets

    if not assets_prefix:
        print("[SIMREADY] ASSETS_PREFIX is required", file=sys.stderr)
        sys.exit(1)

    assets_root = GCS_ROOT / assets_prefix
    scene_assets_path = assets_root / "scene_assets.json"

    print(f"[SIMREADY] Bucket={bucket}")
    print(f"[SIMREADY] Scene={scene_id}")
    print(f"[SIMREADY] Assets root={assets_root}")
    print(f"[SIMREADY] Loading {scene_assets_path}")

    scene_assets = load_json(scene_assets_path)
    objects = scene_assets.get("objects", [])
    print(f"[SIMREADY] Found {len(objects)} objects in scene_assets.json")

    client = None
    if have_gemini():
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("[SIMREADY] Gemini client initialized")
    else:
        print(
            "[SIMREADY] GEMINI_API_KEY not set or google-genai unavailable; using heuristic defaults only",
            file=sys.stderr,
        )

    simready_paths: Dict[Any, str] = {}

    for obj in objects:
        oid = obj.get("id")
        if oid is None:
            continue

        obj_type = obj.get("type")
        if obj_type != "static":
            # For now, treat interactive / other types as already sim-ready.
            print(f"[SIMREADY] Skipping non-static obj {oid} (type={obj_type})")
            continue

        print(f"[SIMREADY] Processing static obj {oid}")

        visual = choose_static_visual_asset(assets_root, oid)
        if visual is None:
            print(f"[SIMREADY] WARNING: no visual asset found for obj {oid}", file=sys.stderr)
            continue

        visual_path, visual_rel = visual
        obj_metadata = load_object_metadata(GCS_ROOT, obj, assets_prefix)
        physics_cfg = call_gemini_for_object(client, oid, obj, obj_metadata)

        # Place simready.usda next to the visual asset.
        sim_dir = visual_path.parent
        ensure_dir(sim_dir)
        sim_path = sim_dir / "simready.usda"

        write_simready_usd(sim_path, visual_rel, physics_cfg)

        sim_rel = f"{assets_prefix}/obj_{oid}/simready.usda"
        if "static/obj_" in str(visual_path):
            sim_rel = f"{assets_prefix}/static/obj_{oid}/simready.usda"

        print(f"[SIMREADY] Wrote simready asset for obj {oid} -> {sim_path}")
        simready_paths[oid] = sim_rel

    # Update scene_assets.json in-place to include simready_usd references.
    if simready_paths:
        updated_objects: List[dict] = []
        for obj in objects:
            oid = obj.get("id")
            if oid in simready_paths:
                obj = dict(obj)
                obj["simready_usd"] = simready_paths[oid]
            updated_objects.append(obj)
        scene_assets["objects"] = updated_objects
        scene_assets_path.write_text(json.dumps(scene_assets, indent=2), encoding="utf-8")
        print(f"[SIMREADY] Updated {scene_assets_path} with simready_usd paths")
    else:
        print(
            "[SIMREADY] No simready assets were created; scene_assets.json left unchanged",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
