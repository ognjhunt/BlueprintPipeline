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


# ---------- Generic fallback (used only when Gemini fails) ----------

# Minimal generic fallback with very wide safety bounds
GENERIC_FALLBACK: Dict[str, Any] = {
    "density_kg_per_m3": 400.0,
    "mass_range_kg": (0.001, 500.0),  # Very wide range: 1g to 500kg
    "friction": 0.7,
    "restitution": 0.2,
    "dynamic": True,
    "collision_shape": "mesh",
    "material_name": "generic",
}


# ---------- Physics config (default + Gemini) ----------

def estimate_default_physics(obj: Dict[str, Any], metadata: Optional[dict]) -> Dict[str, Any]:
    """
    Minimal fallback physics estimate used only when Gemini is unavailable or fails.
    This provides a simple baseline that Gemini will completely override.
    """
    size = extract_mesh_size_from_metadata(metadata)
    volume = None
    if size:
        try:
            sx, sy, sz = [max(float(s), 1e-3) for s in size]
            volume = sx * sy * sz  # m^3
        except Exception:
            volume = None

    rho = float(GENERIC_FALLBACK.get("density_kg_per_m3", 400.0))
    mass: float
    if volume is not None:
        mass_est = volume * rho
        # Use very wide safety bounds
        mass = float(np.clip(mass_est, 0.001, 500.0))
    else:
        # If we know nothing about volume, default to 1kg
        mass = 1.0

    dynamic = bool(GENERIC_FALLBACK.get("dynamic", True))
    friction = float(GENERIC_FALLBACK.get("friction", 0.7))
    restitution = float(GENERIC_FALLBACK.get("restitution", 0.2))
    collision_shape = str(GENERIC_FALLBACK.get("collision_shape", "mesh"))
    material_name = str(GENERIC_FALLBACK.get("material_name", "generic"))

    note_parts = [f"fallback estimate", f"rho={rho:.2f} kg/m^3"]
    if volume is not None:
        note_parts.append(f"volume={volume:.4f} m^3")
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
    Prompt Gemini to estimate realistic physics parameters based on object metadata.
    Uses real-world examples to guide the model but doesn't constrain output.
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
    }

    skeleton = {
        "dynamic": True,
        "collisionShape": "mesh",
        "restitution": 0.2,
        "friction": 0.7,
        "mass_kg": 1.0,
        "density_kg_per_m3": 400.0,
        "material_name": "generic",
        "notes": "",
    }

    prompt = f"""
You are helping configure 3D assets for robotics training in NVIDIA Isaac Sim / USD.

Your task is to estimate REALISTIC physics parameters for this object based on its type,
name, and dimensions. Use your knowledge and grounding capabilities to look up real-world
physics properties for the specific object.

The simulation uses:
- meters for linear distance
- kilograms for mass
- rigid bodies only (no joints in this step)

CRITICAL: Provide accurate real-world physics for THIS SPECIFIC OBJECT TYPE.
Do NOT use generic values. Research and apply realistic properties.

## Examples of Real-World Physics (for reference):

**Utensils:**
- Spoon (tablespoon): ~0.02-0.04 kg, stainless steel, friction ~0.4
- Fork (dinner fork): ~0.03-0.05 kg, stainless steel, friction ~0.4
- Knife (dinner knife): ~0.04-0.08 kg, stainless steel, friction ~0.35
- Spatula: ~0.05-0.15 kg, metal/plastic, friction ~0.5

**Kitchen Items:**
- Mug (ceramic): ~0.25-0.4 kg, ceramic, friction ~0.6
- Plate (dinner): ~0.3-0.6 kg, ceramic, friction ~0.6
- Bowl: ~0.2-0.5 kg, ceramic/glass, friction ~0.6
- Microwave: ~12-20 kg, metal/plastic, friction ~0.6

**Textiles:**
- Blanket (throw): ~0.5-2 kg, fabric, friction ~1.3
- Towel: ~0.2-0.6 kg, fabric, friction ~1.2
- Pillow: ~0.3-1 kg, fabric, friction ~1.2

**Furniture:**
- Chair (dining): ~5-15 kg, wood, friction ~0.8
- Sofa: ~30-80 kg, upholstery, friction ~0.9
- Table (coffee): ~10-30 kg, wood, friction ~0.8

**Electronics:**
- Laptop: ~1-3 kg, metal/plastic, friction ~0.5
- Phone: ~0.15-0.25 kg, glass/metal, friction ~0.6
- Tablet: ~0.3-0.7 kg, glass/metal, friction ~0.6

**Small Items:**
- Book: ~0.3-1 kg, paper, friction ~0.9
- Hat: ~0.05-0.2 kg, fabric, friction ~1.2
- Shoe: ~0.3-0.8 kg, rubber/leather, friction ~1.1

## Physics Parameters Guide:

**Mass (mass_kg):**
- Use real-world mass for this specific object type
- Consider the object's size and typical materials
- Range: 0.001 kg (very light) to 500 kg (very heavy furniture/appliances)

**Friction:**
- Rubber/fabric: 1.0-1.5 (high grip)
- Wood: 0.7-0.9
- Ceramic/glass: 0.5-0.7
- Metal (steel): 0.3-0.6
- Plastic: 0.4-0.7
- Range: 0.0 (ice-like) to 2.0 (maximum)

**Restitution (bounciness):**
- Most household objects: 0.05-0.2 (minimal bounce)
- Rubber balls: 0.7-0.9
- Range: 0.0 (no bounce) to 1.0 (perfect bounce)

**Collision Shape:**
- "box": for rectangular furniture, books, boxes
- "mesh": for irregular shapes, utensils, complex objects (default)
- "sphere": for balls
- "capsule": for cylindrical objects

**Material Name:**
- Use descriptive names: "stainless_steel", "ceramic", "fabric", "wood",
  "plastic", "glass", "rubber", "metal_plastic", "upholstery", etc.

**Density (density_kg_per_m3):**
- Calculate as: mass_kg / bounding_box_volume_m3
- This is effective bulk density (includes air gaps)

**Dynamic:**
- true for movable objects (almost everything)
- false only for fixed architectural elements (walls, floors)

Return ONLY valid JSON (no markdown, no comments, no extra text) that matches this structure:

{json.dumps(skeleton, indent=2)}

Here is the object metadata:

{json.dumps(minimal, indent=2)}

Estimate realistic physics parameters for this specific object based on its name, type,
and size. Use your knowledge and grounding to provide accurate real-world values.
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
    Ask Gemini to estimate realistic physics parameters for the object.
    No category constraints - Gemini has full control over all physics parameters.
    """
    base_cfg = estimate_default_physics(obj, metadata)

    if client is None:
        return base_cfg

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

        # Merge Gemini's response onto fallback defaults
        merged = dict(base_cfg)

        # Helper to safely extract and validate floats with WIDE safety bounds
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

        # Friction/restitution: keep reasonable physics bounds
        merged["friction"] = _get_clamped(
            "friction", merged.get("friction", 0.7), 0.0, 2.0
        )
        merged["restitution"] = _get_clamped(
            "restitution", merged.get("restitution", 0.2), 0.0, 1.0
        )

        # Mass: Use WIDE safety bounds - let Gemini determine realistic values
        # Only prevent obviously broken values (< 1g or > 1000kg)
        mass_default = float(merged.get("mass_kg", 1.0))
        merged["mass_kg"] = _get_clamped(
            "mass_kg", mass_default, 0.001, 1000.0
        )

        # Density: Accept wide range, just prevent unrealistic extremes
        try:
            rho_val = float(data.get("density_kg_per_m3", merged.get("density_kg_per_m3", 400.0)))
            if rho_val > 0.0 and rho_val < 10000.0:  # Up to dense metals
                merged["density_kg_per_m3"] = rho_val
        except Exception:
            pass

        if "material_name" in data and isinstance(data["material_name"], str):
            merged["material_name"] = data["material_name"]

        if "notes" in data and isinstance(data["notes"], str):
            merged["notes"] = data["notes"]

        # Mark that Gemini estimated this
        merged["notes"] = (merged.get("notes", "") + " | gemini_estimated").strip()

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
