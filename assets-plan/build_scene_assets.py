import os, json, sys
from pathlib import Path

from tools.validation.entrypoint_checks import validate_required_env_vars
INTERACTIVE_KEYWORDS = {
    "door", "drawer", "cabinet", "fridge", "refrigerator", "oven", "freezer", "handle", "knob",
    "faucet", "tap", "switch", "lever", "hinge"
}

def parse_interactive_ids(raw: str | None):
    if not raw:
        return set()
    ids = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.add(int(tok))
        except ValueError:
            pass
    return ids

def classify_type(class_name: str, phrase: str | None, interactive_ids, obj_id, static_pipeline: str, sim_role: str | None = None, skip_interactive: bool = False):
    # If SKIP_INTERACTIVE_JOB is enabled, force all objects to be static
    if skip_interactive:
        return "static", static_pipeline

    # Special handling: scene_background is always static (it's the room shell)
    if class_name == "scene_background" or obj_id == "scene_background":
        return "static", static_pipeline

    # Special handling: scene_shell sim_role is always static
    if sim_role == "scene_shell":
        return "static", static_pipeline

    # First check sim_role (most authoritative)
    if sim_role in ("articulated_furniture", "articulated_appliance", "manipulable_object"):
        return "interactive", "particulate"

    # Fallback to manual interactive_ids
    text = (class_name or "").lower()
    if phrase:
        text += " " + phrase.lower()
    if obj_id in interactive_ids:
        return "interactive", "particulate"
    if any(k in text for k in INTERACTIVE_KEYWORDS):
        return "interactive", "particulate"
    return "static", static_pipeline


def infer_objects_from_multiview(multiview_root: Path, multiview_prefix: str):
    """
    Build object entries by scanning multiview outputs when the layout is empty.

    We look for directories named `obj_*` (except `obj_scene_background`) and use
    their generation metadata when available.
    """

    inferred_objects = []

    for mv_dir in sorted(multiview_root.glob("obj_*")):
        if not mv_dir.is_dir():
            continue

        obj_stub = mv_dir.name.removeprefix("obj_")
        if obj_stub == "scene_background":
            continue

        meta_path = mv_dir / "generation_meta.json"
        obj_id = obj_stub
        class_name = obj_stub
        phrase = None

        sim_role = None
        if meta_path.is_file():
            try:
                with meta_path.open("r") as f:
                    meta = json.load(f)
                obj_id = meta.get("object_id", obj_id)
                class_name = meta.get("category") or meta.get("class_name") or class_name
                phrase = meta.get("short_description")
                sim_role = meta.get("sim_role")
            except Exception as e:  # pragma: no cover - defensive logging
                print(f"[ASSETS] WARNING: Failed to read {meta_path}: {e}; using defaults")

        # Mirror the layout object structure enough for downstream logic
        inferred_objects.append({
            "id": obj_id,
            "class_name": class_name,
            "object_phrase": phrase,
            "sim_role": sim_role,
        })

    if inferred_objects:
        print(f"[ASSETS] Inferred {len(inferred_objects)} objects from multiview outputs under {multiview_prefix}")
    else:
        print(f"[ASSETS] No multiview objects found under {multiview_prefix}; keeping plan empty")

    return inferred_objects

def main():
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
            "MULTIVIEW_PREFIX": "Path prefix for multiview inputs (scenes/<sceneId>/multiview)",
            "ASSETS_PREFIX": "Path prefix for assets (scenes/<sceneId>/assets)",
        },
        label="[ASSETS]",
    )

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    layout_prefix = os.getenv("LAYOUT_PREFIX")       # scenes/<sceneId>/layout
    multiview_prefix = os.environ["MULTIVIEW_PREFIX"] # scenes/<sceneId>/multiview
    assets_prefix = os.environ["ASSETS_PREFIX"]       # scenes/<sceneId>/assets
    seg_prefix = os.getenv("SEG_PREFIX")             # scenes/<sceneId>/seg (optional, for Gemini pipeline)
    layout_file = os.getenv("LAYOUT_FILE_NAME", "scene_layout_scaled.json")
    interactive_ids_env = os.getenv("INTERACTIVE_OBJECT_IDS", "")
    static_pipeline = os.getenv("STATIC_ASSET_PIPELINE", "ultrashape")
    particulate_endpoint = os.getenv("PARTICULATE_ENDPOINT")
    skip_interactive = os.getenv("SKIP_INTERACTIVE_JOB", "0").lower() in ("1", "true", "yes", "on")

    if not (multiview_prefix and assets_prefix):
        raise SystemExit("[ASSETS] MULTIVIEW_PREFIX, ASSETS_PREFIX required")

    interactive_ids = parse_interactive_ids(interactive_ids_env)

    root = Path("/mnt/gcs")

    # Try to find layout file in multiple locations
    # 1. LAYOUT_PREFIX (standard pipeline)
    # 2. SEG_PREFIX (Gemini pipeline)
    layout_path = None
    search_paths = []

    if layout_prefix:
        candidate = root / layout_prefix / layout_file
        search_paths.append(candidate)
        if candidate.is_file():
            layout_path = candidate

    if layout_path is None and seg_prefix:
        candidate = root / seg_prefix / layout_file
        search_paths.append(candidate)
        if candidate.is_file():
            layout_path = candidate

    if layout_path is None:
        print(f"[ASSETS] ERROR: Could not find {layout_file} in any of these locations:", file=sys.stderr)
        for p in search_paths:
            print(f"[ASSETS]   - {p}", file=sys.stderr)
        raise SystemExit("[ASSETS] Layout file not found")

    multiview_root = root / multiview_prefix
    assets_root = root / assets_prefix

    print(f"[ASSETS] Bucket={bucket}")
    print(f"[ASSETS] Scene={scene_id}")
    print(f"[ASSETS] Layout={layout_path}")
    print(f"[ASSETS] Multiview root={multiview_root}")
    print(f"[ASSETS] Assets root={assets_root}")
    print(f"[ASSETS] interactive_ids={interactive_ids}")
    print(f"[ASSETS] static_pipeline={static_pipeline}")
    print(f"[ASSETS] skip_interactive={skip_interactive}")
    if particulate_endpoint:
        print(f"[ASSETS] particulate_endpoint={particulate_endpoint}")

    with layout_path.open("r") as f:
        layout = json.load(f)

    objects = layout.get("objects") or []
    if not objects:
        print("[ASSETS] Layout contains 0 objects; attempting to infer from multiview outputs")
        objects = infer_objects_from_multiview(multiview_root, multiview_prefix)
    plan_objects = []

    for obj in objects:
        oid = obj.get("id")
        cls = obj.get("class_name", f"class_{obj.get('class_id', 0)}")
        phrase = obj.get("object_phrase")  # optional future field
        sim_role = obj.get("sim_role")  # from Gemini inventory
        approx_location = obj.get("approx_location")  # from Gemini inventory

        mv_dir = multiview_root / f"obj_{oid}"

        # Support both crop.png (from crop-based multiview) and view_0.png (from gemini-generative multiview)
        crop_path = mv_dir / "crop.png"
        view_path = mv_dir / "view_0.png"

        if crop_path.is_file():
            image_path = crop_path
        elif view_path.is_file():
            image_path = view_path
        else:
            print(f"[ASSETS] WARNING: missing crop.png or view_0.png for obj {oid} in {mv_dir}")
            continue

        obj_type, pipeline = classify_type(cls, phrase, interactive_ids, oid, static_pipeline=static_pipeline, sim_role=sim_role, skip_interactive=skip_interactive)

        # Use the relative path for the image that was found
        image_filename = image_path.name

        entry = {
            "id": oid,
            "class_name": cls,
            "type": obj_type,
            "pipeline": pipeline,
            "multiview_dir": f"{multiview_prefix}/obj_{oid}",
            "crop_path": f"{multiview_prefix}/obj_{oid}/{image_filename}",
            "polygon": obj.get("polygon"),
        }
        # Carry forward approx_location for synthetic position generation
        if approx_location:
            entry["approx_location"] = approx_location
        if obj_type == "interactive":
            entry["interactive_output"] = f"{assets_prefix}/interactive/obj_{oid}"
            if particulate_endpoint:
                entry["particulate_endpoint"] = particulate_endpoint
        else:
            entry["asset_path"] = f"{assets_prefix}/obj_{oid}/asset.glb"
        plan_objects.append(entry)

    assets_root.mkdir(parents=True, exist_ok=True)
    out_path = assets_root / "scene_assets.json"
    plan = {
        "scene_id": scene_id,
        "objects": plan_objects,
        "particulate_endpoint": particulate_endpoint,
        "interactive_count": sum(1 for o in plan_objects if o["type"] == "interactive"),
    }
    with out_path.open("w") as f:
        json.dump(plan, f, indent=2)

    print(f"[ASSETS] Wrote {out_path} with {len(plan_objects)} objects")

if __name__ == "__main__":
    main()
