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
    if not metadata:
        return None
    mesh_bounds = metadata.get("mesh_bounds") or {}
    export_bounds = mesh_bounds.get("export") or mesh_bounds.get("bounds") or mesh_bounds
    size = export_bounds.get("size")
    if size and len(size) == 3:
        return [float(s) for s in size]
    return None


# ---------- Physics config (default + Gemini) ----------

def estimate_default_physics(obj: Dict[str, Any], metadata: Optional[dict]) -> Dict[str, Any]:
    """
    Cheap heuristic fall-back if Gemini is unavailable:
    - mass from approximate volume * generic density.
    """
    size = extract_mesh_size_from_metadata(metadata)
    volume = None
    if size:
        sx, sy, sz = [max(float(s), 1e-3) for s in size]
        volume = sx * sy * sz  # m^3

    # "Generic household object" density ~ 400 kg/m^3
    density = 400.0
    if volume is not None:
        mass = max(volume * density, 0.05)
    else:
        # If we know nothing, guess 1 kg
        mass = 1.0

    return {
        "dynamic": True,
        "collisionShape": "mesh",
        "restitution": 0.2,
        "friction": 0.9,
        "mass_kg": float(mass),
        "density_kg_per_m3": float(density),
        "material_name": "generic",
        "notes": "",
    }


def make_gemini_prompt(oid: Any, obj: Dict[str, Any], metadata: Optional[dict]) -> str:
    minimal = {
        "id": oid,
        "type": obj.get("type"),
        "class_name": obj.get("class_name"),
        "pipeline": obj.get("pipeline"),
        "obb_extents": (obj.get("obb") or {}).get("extents"),
        "mesh_metadata": metadata or {},
    }

    skeleton = {
        "dynamic": True,
        "collisionShape": "mesh",
        "restitution": 0.2,
        "friction": 0.9,
        "mass_kg": 1.0,
        "density_kg_per_m3": 400.0,
        "material_name": "generic",
        "notes": "",
    }

    prompt = f"""
You are helping configure 3D assets for robotics training in NVIDIA Isaac Sim / USD.

Given object metadata, propose plausible physics parameters that make the object
behave realistically in simulation (for grasping, pushing, etc.).

Constraints:
- Units are meters (size) and kilograms (mass).
- Assume a single rigid body, no joints.
- Prefer conservative values (not super bouncy, not frictionless).
- If uncertain, choose reasonable defaults.

Return ONLY valid JSON (no comments, no extra text) that matches this structure:

{skeleton!r}

Meaning of fields:
- dynamic: true if the object should move under physics (almost everything except walls/floor).
- collisionShape: one of "box", "sphere", "capsule", "mesh".
- restitution: 0 (no bounce) to 1 (very bouncy).
- friction: 0 (very slippery) to 2 (very sticky).
- mass_kg: positive float.
- density_kg_per_m3: positive float, approximate bulk material density.
- material_name: short label like "wood", "metal", "plastic", "rubber".
- notes: very short explanation of your choices.

Here is the metadata for this object:

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
    base_cfg = estimate_default_physics(obj, metadata)

    if client is None:
        return base_cfg

    prompt = make_gemini_prompt(oid, obj, metadata)

    try:
        # Ask for pure JSON back
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=getattr(types, "ThinkingConfig", None)(
                    thinking_level="HIGH"
                ) if hasattr(types, "ThinkingConfig") else None,
            )
        except TypeError:
            # Older versions may not accept thinking_config
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )

        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.0-pro-preview"),
            contents=prompt,
            config=config,
        )

        raw = response.text or ""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")

        # Merge onto defaults; ignore unknown keys.
        merged = dict(base_cfg)
        for k in merged.keys():
            if k in data and data[k] is not None:
                merged[k] = data[k]
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
    Create a tiny USD wrapper that:
    - makes the object a rigid body
    - adds mass
    - adds a collision API
    - references the visual asset stage.
    """
    mass = float(physics.get("mass_kg", 1.0))
    dynamic = bool(physics.get("dynamic", True))

    # Build the token string separately to avoid backslashes inside the f-string
    enabled_token = "true" if dynamic else "false"

    lines: List[str] = []
    lines.append("#usda 1.0")
    lines.append("(\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n)")
    lines.append("")
    lines.append('def Xform "Asset" (')
    lines.append('    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]')
    lines.append(")")
    lines.append("{")
    lines.append(f"    float physics:mass = {mass:.6f}")
    # Simple flag; Isaac Sim will respect this for dynamic bodies.
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
